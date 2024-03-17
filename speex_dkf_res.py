# The echo canceller is based on the MDF algorithm described in:
#
# J. S. Soo, K. K. Pang Multidelay block frequency adaptive filter,
# IEEE Trans. Acoust. Speech Signal Process., Vol. ASSP-38, No. 2,
# February 1990.
#
# We use the Alternatively Updated MDF (AUMDF) variant. Robustness to
# double-talk is achieved using a variable learning rate as described in:
#
# Valin, J.-M., On Adjusting the Learning Rate in Frequency Domain Echo
# Cancellation With Double-Talk. IEEE Transactions on Audio,
# Speech and Language Processing, Vol. 15, No. 3, pp. 1030-1034, 2007.
# http://people.xiph.org/~jm/papers/valin_taslp2006.pdf
#
# There is no explicit double-talk detection, but a continuous variation
# in the learning rate based on residual echo, double-talk and background
# noise.
#
# Another kludge that seems to work good: when performing the weight
# update, we only move half the way toward the "goal" this seems to
# reduce the effect of quantization noise in the update phase. This
# can be seen as applying a gradient descent on a "soft constraint"
# instead of having a hard constraint.
#
# Notes for this file:
#
# Usage:
#
#    processor = MDF(Fs, frame_size, filter_length)
#    processor.main_loop(u, d)
#
#    Fs                  sample rate
#    u                   speaker signal, vector in range [-1; 1]
#    d                   microphone signal, vector in range [-1; 1]
#    filter_length       typically 250ms, i.e. 4096 @ 16k FS
#                        must be a power of 2
#    frame_size          typically 8ms, i.e. 128 @ 16k Fs
#                        must be a power of 2
#
# Shimin Zhang <shmzhang@npu-aslp.org>
#
import wave

import numpy as np
from scipy.io import wavfile
from scipy import signal

def float_to_short(x):
    x = x * 32768.0
    #x = np.round(x)
    x[x < -32767.5] = -32768
    x[x > 32766.5] = 32767
    x = np.floor(x+0.5)
    return x


class MDF:
    def __init__(self, fs: int, frame_size: int, filter_length: int) -> None:
        nb_mic = 1
        nb_speakers = 1
        self.K = nb_speakers
        K = self.K
        self.C = nb_mic
        C = self.C
        self.A = 0.999
        self.A2 = self.A * self.A
        self.frame_size = frame_size
        self.filter_length = filter_length
        self.window_size = frame_size * 2
        N = self.window_size
        self.M = int(np.fix((filter_length + frame_size - 1) / frame_size))
        M = self.M
        self.cancel_count = -1
        self.sum_adapt = 0
        self.saturated = 0
        self.screwed_up = 0

        self.sampling_rate = fs
        self.spec_average = (self.frame_size) / (self.sampling_rate)
        self.beta0 = (2.0 * self.frame_size) / self.sampling_rate
        self.beta_max = (.5 * self.frame_size) / self.sampling_rate
        self.leak_estimate = 0

        self.e = np.zeros((N, C), )
        self.x = np.zeros((N, K), )
        self.input = np.zeros((self.frame_size, C), )
        self.y = np.zeros((N, C), )
        self.last_y = np.zeros((N, C), )
        self.Yf = np.zeros((self.frame_size + 1, 1), )
        self.Rf = np.zeros((self.frame_size + 1, 1), )
        self.Xf = np.zeros((self.frame_size + 1, 1), )
        self.Yh = np.zeros((self.frame_size + 1, 1), )
        self.Eh = np.zeros((self.frame_size + 1, 1), )
        self.P = np.full((N, M), 10, dtype=np.float64)
        self.X = np.zeros((N, K, M), dtype=np.complex)
        self.Y = np.zeros((N, C), dtype=np.complex)
        self.E = np.zeros((N, C), dtype=np.complex)
        self.W = np.zeros((N, K, M, C), dtype=np.complex)
        self.foreground = np.zeros((N, K, M, C), dtype=np.complex)
        self.PHI = np.zeros((frame_size + 1, 1), )
        self.power = np.zeros((frame_size + 1, 1), )
        self.power_1 = np.ones((frame_size + 1, 1), )
        self.window = np.zeros((N, 1), )
        self.prop = np.zeros((M, 1), )
        self.wtmp = np.zeros((N, 1), )
        self.window = .5 - .5 * \
                      np.cos(2 * np.pi * (np.arange(1, N + 1).reshape(-1, 1) - 1) / N)
        decay = np.exp(-2.4 / M)
        self.prop[0, 0] = .7
        for i in range(1, M):
            self.prop[i, 0] = self.prop[i - 1, 0] * decay
        self.prop = (.8 * self.prop) / np.sum(self.prop)

        self.memX = np.zeros((K, 1), )
        self.memD = np.zeros((C, 1), )
        self.memE = np.zeros((C, 1), )
        self.preemph = .9
        if self.sampling_rate < 12000:
            self.notch_radius = .9
        elif self.sampling_rate < 24000:
            self.notch_radius = .982
        else:
            self.notch_radius = .992
        self.notch_mem = np.zeros((2 * C, 1), )
        self.adapted = 1
        self.Pey = 1
        self.Pyy = 1
        self.Davg1 = 0
        self.Davg2 = 0
        self.Dvar1 = 0
        self.Dvar2 = 0

        # 后处理的初始化
        self.window2 = np.zeros((2*self.frame_size, 1), )
        self.ps_size = self.frame_size
        self.ns_level = 1
        self.denoise_enable = 1
        self.vad_enable = 0
        self.dereverb_enable = 0
        self.reverb_decay = 0
        self.reverb_level = 0
        if self.ns_level == 0:
            self.noise_suppress = -5
        elif self.ns_level == 2:
            self.noise_suppress = -25
        else:
            self.noise_suppress = -15
        self.echo_suppress = -45
        self.echo_suppress_active = -15
        self.speech_prob_start = 0.35
        self.speech_prob_continue = 0.2
        self.nbands = 24
        self.M_2 = self.nbands
        self.nb_adapt = 0
        self.min_count = 0
        self.echo_state = 1# 前处理指针，这里用flag代替
        self.do_dereverb = 0
        self.residual_echo = np.zeros((self.frame_size+1, 1),)
        self.echo_noise = np.zeros((self.frame_size+self.nbands, 1), )

        df = self.sampling_rate / (2*self.frame_size)
        # (13.1f*atan(.00074f*(n))+2.24f*atan((n)*(n)*1.85e-8f)+1e-4f*(n))
        max_mel = 13.1 * np.arctan(.00074*(self.sampling_rate/2))+2.24*np.arctan(self.sampling_rate/2*self.sampling_rate/2*1.85e-8)+1e-4*self.sampling_rate/2
        mel_interval = max_mel / (self.nbands-1)
        self.bank_left = np.zeros((self.frame_size, 1),)
        self.bank_right = np.zeros((self.frame_size, 1), )
        self.filter_left = np.zeros((self.frame_size, 1), )
        self.filter_right = np.zeros((self.frame_size, 1), )
        self.scaling = np.zeros((self.nbands, 1),)
        for i in range(self.frame_size):
            curr_freq = i*df
            mel = 13.1 * np.arctan(.00074*curr_freq)+2.24*np.arctan(curr_freq*curr_freq*1.85e-8)+1e-4*curr_freq
            if mel > max_mel:
                break
            id1 = int(np.floor(mel/mel_interval))
            if id1 > self.nbands - 2:
                id1 = self.nbands - 2
                val = 1.0
            else:
                val = (mel-id1*mel_interval)/mel_interval
            id2 = id1 +1
            self.bank_left[i] = id1
            self.filter_left[i] = 1 - val
            self.bank_right[i] = id2
            self.filter_right[i] = val
            for i in range(self.nbands):
                self.scaling[i] = 0
            for i in range(self.frame_size):
                id = self.bank_left[i]
                self.scaling[int(id)] += self.filter_left[i]
                id = self.bank_right[i]
                self.scaling[int(id)] += self.filter_right[i]
            for i in range(self.nbands):
                self.scaling[i] = 1 / (self.scaling[i]+1e-8)

            self.frame = np.zeros((2*self.ps_size, 1),)
            N3 = 2*self.ps_size - self.frame_size
            N4 = self.frame_size - N3
            self.inbuf = np.zeros((N3, 1), )
            self.outbuf = np.zeros((N3, 1), )
            self.ps = np.zeros((self.frame_size+self.nbands, 1), )
            self.S = np.zeros((self.frame_size, 1),)
            self.update_prob = np.ones((self.frame_size, 1),)
            self.noise = np.ones((self.frame_size+self.nbands),)
            self.reverb_estimation = np.zeros((self.frame_size+self.nbands), )
            self.old_ps = np.ones((self.frame_size+self.nbands),)
            self.gain = np.ones((self.frame_size+self.nbands,1),)
            self.post = np.ones((self.frame_size+self.nbands,1),)
            self.prior = np.ones((self.frame_size+self.nbands,1),)

            self.zeta = np.zeros((self.frame_size+self.nbands, 1), )
            self.gain_floor = np.zeros((self.frame_size+self.nbands, 1),)
            self.gain2 = np.zeros((self.frame_size+self.nbands, 1), )

    def main_loop(self, u, d):
        """MDF core function

        Args:
            u (array): reference signal
            d (array): microphone signal
        """
        assert u.shape == d.shape
        u = float_to_short(u)
        d = float_to_short(d)

        e = np.zeros_like(u)
        y = np.zeros_like(u)
        end_point = len(u)

        for n in range(0, end_point, self.frame_size):
            nStep = np.floor(n / self.frame_size) + 1
            self.nStep = nStep
            # the break operation not understand.
            # only for signal channel AEC
            if n + self.frame_size > end_point:
                break
            u_frame = u[n:n + self.frame_size]
            d_frame = d[n:n + self.frame_size]
            out = self.speex_echo_cancellation_mdf(d_frame[:, None], u_frame[:, None])[:, 0]
            out = self.speex_r2_preprocess_run(out)
            e[n:n + self.frame_size] = out
            y[n:n + self.frame_size] = d_frame - out
            print(f"epoch {int(n / self.frame_size)} finished")
        e = e / 32768.0
        y = y / 32768.0
        return e, y

    def speex_echo_cancellation_mdf(self, mic, far_end):
        N = self.window_size
        M = self.M
        C = self.C
        K = self.K

        Pey_cur = 1
        Pyy_cur = 1

        out = np.zeros((self.frame_size, C), )
        self.cancel_count += 1

        ss = .35 / M
        ss_1 = 1 - ss

        for chan in range(C):
            # Apply a notch filter to make sure DC doesn't end up causing problems
            self.input[:, chan], self.notch_mem[:, chan] = self.filter_dc_notch16(
                mic[:, chan], self.notch_mem[:, chan])

            for i in range(self.frame_size):
                tmp32 = self.input[i, chan] - \
                        (np.dot(self.preemph, self.memD[chan]))
                self.memD[chan] = self.input[i, chan]
                self.input[i, chan] = tmp32

        for speak in range(K):
            for i in range(self.frame_size):
                self.x[i, speak] = self.x[i + self.frame_size, speak]
                tmp32 = far_end[i, speak] - \
                        np.dot(self.preemph, self.memX[speak])
                self.x[i + self.frame_size, speak] = tmp32
                self.memX[speak] = far_end[i, speak]

        # self.X = np.roll(self.X, [0, 0, 1])
        self.X = np.roll(self.X, 1, axis=2)

        for speak in range(K):
            self.X[:, speak, 0] = np.fft.fft(self.x[:, speak]) / N
            self.X[0, speak, 0] = 0

        Sxx = 0
        for speak in range(K):
            Sxx = Sxx + np.sum(self.x[self.frame_size:, speak] ** 2)
            self.Xf = np.abs(self.X[:self.frame_size + 1, speak, 0]) ** 2
        Sff = 0
        self.Y = np.zeros((N, C), dtype=np.complex)
        for chan in range(C):
            self.Y[:, chan] = 0
            for speak in range(K):
                for j in range(M):
                    self.Y[:, chan] = self.Y[:, chan] + self.X[:,
                                                        speak, j] * self.foreground[:, speak, j, chan]
            self.e[:, chan] = np.fft.ifft(self.Y[:, chan]).real * N
            self.e[:self.frame_size, chan] = self.input[:, chan] - \
                                             self.e[self.frame_size:, chan]
            Sff = Sff + np.sum(np.abs(self.e[:self.frame_size, chan]) ** 2)

        if self.adapted:
            self.prop = self.mdf_adjust_prop()
        if self.saturated == 0:
            #X2 = np.sum(np.abs(self.X) ** 2, axis=0)
            for chan in range(C):
                e_fft = np.zeros(shape=(self.frame_size*2,), dtype=np.float64)
                e_fft[self.frame_size:] = self.e[:self.frame_size, chan]
                E = np.fft.fft(e_fft)/ N
                E[0] = 0
                for speak in range(K):
                    X2 = np.expand_dims(np.sum(np.abs(self.X[:, speak, :]) ** 2, axis=1), 1)
                    Pe = 0.5 * self.P * X2 + np.expand_dims(np.abs(E) ** 2 / self.M, 1)
                    mu = 0.5 * self.P / (Pe + 1e-8)
                    self.P = self.A2 * (1 - 0.5 * mu * X2) * self.P + (1 - self.A2) * np.abs(self.W[:, speak, :, chan])** 2
                    G = mu * self.X[:, speak, :].conj()
                    #self.W[:, speak, :, chan] += np.transpose(self.prop).repeat(N, axis=0) * np.expand_dims(E, 1) * G
                    self.W[:, speak, :, chan] += 0.5 * np.expand_dims(E, 1) * G

        else:
            self.saturated -= 1

        for chan in range(C):
            for speak in range(K):
                for j in range(M):
                    if (self.cancel_count) % M == j:
                        self.W[0, speak, j, chan] = 0
                        self.wtmp = (np.fft.ifft(self.W[:, speak, j, chan]).real)*512
                        self.wtmp[self.frame_size:N] = 0
                        self.W[:, speak, j, chan] = np.fft.fft(self.wtmp)/512
                        self.W[0, speak, j, chan] = 0

        self.Yf = np.zeros((self.frame_size + 1, 1), )
        self.Rf = np.zeros((self.frame_size + 1, 1), )
        self.Xf = np.zeros((self.frame_size + 1, 1), )

        Dbf = 0
        for chan in range(C):
            self.Y[:, chan] = 0
            for speak in range(K):
                for j in range(M):
                    self.Y[:, chan] = self.Y[:, chan] + \
                                      self.X[:, speak, j] * self.W[:, speak, j, chan]
            self.y[:, chan] = np.fft.ifft(self.Y[:, chan]).real * N

        See = 0

        for chan in range(C):
            self.e[:self.frame_size, chan] = self.e[self.frame_size:N,
                                             chan] - self.y[self.frame_size:N, chan]

            Dbf = Dbf + 10 + np.sum(np.abs(self.e[:self.frame_size, chan]) ** 2)
            self.e[:self.frame_size, chan] = self.input[:, chan] - \
                                             self.y[self.frame_size:N, chan]
            See = See + np.sum(np.abs(self.e[:self.frame_size, chan]) ** 2)

        VAR1_UPDATE = .5
        VAR2_UPDATE = .25
        VAR_BACKTRACK = 4
        MIN_LEAK = .005

        self.Davg1 = .6 * self.Davg1 + .4 * (Sff - See)
        self.Dvar1 = .36 * self.Dvar1 + .16 * Sff * Dbf
        self.Davg2 = .85 * self.Davg2 + .15 * (Sff - See)
        self.Dvar2 = .7225 * self.Dvar2 + .0225 * Sff * Dbf

        update_foreground = 0
        if (Sff - See) * abs(Sff - See) > (Sff * Dbf):
            update_foreground = 1
        elif (self.Davg1 * abs(self.Davg1) > (VAR1_UPDATE * self.Dvar1)):
            update_foreground = 1
        elif (self.Davg2 * abs(self.Davg2) > (VAR2_UPDATE * (self.Dvar2))):
            update_foreground = 1

        if update_foreground:
            self.Davg1 = 0
            self.Davg2 = 0
            self.Dvar1 = 0
            self.Dvar2 = 0
            self.foreground = self.W
            for chan in range(C):
                self.e[self.frame_size:N, chan] = (self.window[self.frame_size:N][:, 0] * self.e[self.frame_size:N,
                                                                                          chan]) + (
                                                          self.window[:self.frame_size][:, 0] * self.y[
                                                                                                self.frame_size:N,
                                                                                                chan])
        else:
            reset_background = 0
            if (-(Sff - See) * np.abs(Sff - See) > VAR_BACKTRACK * (Sff * Dbf)):
                reset_background = 1
            if ((-self.Davg1 * np.abs(self.Davg1)) > (VAR_BACKTRACK * self.Dvar1)):
                reset_background = 1
            if ((-self.Davg2 * np.abs(self.Davg2)) > (VAR_BACKTRACK * self.Dvar2)):
                reset_background = 1

            if reset_background:
                self.W = self.foreground
                for chan in range(C):
                    self.y[self.frame_size:N,
                    chan] = self.e[self.frame_size:N, chan]
                    self.e[:self.frame_size, chan] = self.input[:,
                                                     chan] - self.y[self.frame_size:N, chan]
                See = Sff
                self.Davg1 = 0
                self.Davg2 = 0
                self.Dvar1 = 0
                self.Dvar2 = 0

        Sey = 0
        Syy = 0
        Sdd = 0

        for chan in range(C):
            for i in range(self.frame_size):
                tmp_out = self.input[i, chan] - self.e[i + self.frame_size, chan]
                tmp_out = tmp_out + self.preemph * self.memE[chan]
                # This is an arbitrary test for saturation in the microphone signal
                if mic[i, chan] <= -32000 or mic[i, chan] >= 32000:
                    if self.saturated == 0:
                        self.saturated = 1
                out[i, chan] = tmp_out[0]
                self.memE[chan] = tmp_out

            self.e[self.frame_size:N, chan] = self.e[:self.frame_size, chan]
            self.e[:self.frame_size, chan] = 0
            Sey = Sey + np.sum(self.e[self.frame_size:N, chan]
                               * self.y[self.frame_size:N, chan])
            Syy = Syy + np.sum(self.y[self.frame_size:N, chan] ** 2)
            Sdd = Sdd + np.sum(self.input ** 2)

            self.E = np.fft.fft(self.e, axis=0) / N
            self.E[0] = 0
            self.y[:self.frame_size, chan] = 0
            self.Y = np.fft.fft(self.y, axis=0) / N
            self.Y[0] = 0
            self.Rf = np.abs(self.E[:self.frame_size + 1, chan]) ** 2
            self.Yf = np.abs(self.Y[:self.frame_size + 1, chan]) ** 2
        if not (Syy >= 0 and Sxx >= 0 and See >= 0):
            self.screwed_up = self.screwed_up + 50
            out = np.zeros_like(out)
        elif Sff > Sdd + N * 10000:
            self.screwed_up = self.screwed_up + 1
        else:
            self.screwed_up = 0
        if self.screwed_up >= 50:
            print("Screwed up, full reset")
            self.__init__(self.sampling_rate,
                          self.frame_size, self.filter_length)

        See = max(See, N * 100)
        for speak in range(K):
            Sxx = Sxx + np.sum(self.x[self.frame_size:, speak] ** 2)
            self.Xf = np.abs(self.X[:self.frame_size + 1, speak, 0]) ** 2
        self.power = ss_1 * self.power + 1 + ss * self.Xf[:, None]
        Eh_cur = self.Rf - self.Eh.squeeze(1)
        Yh_cur = self.Yf - self.Yh.squeeze(1)
        Pey_cur = Pey_cur + np.sum(Eh_cur * Yh_cur)
        Pyy_cur = Pyy_cur + np.sum(Yh_cur ** 2)
        self.Eh = (1 - self.spec_average) * self.Eh + np.expand_dims(self.spec_average * self.Rf, 1)
        self.Yh = (1 - self.spec_average) * self.Yh + np.expand_dims(self.spec_average * self.Yf, 1)
        Pyy = np.sqrt(Pyy_cur)
        Pey = Pey_cur / Pyy
        tmp32 = self.beta0 * Syy
        if tmp32 > self.beta_max * See:
            tmp32 = self.beta_max * See
        alpha = tmp32 / See
        alpha_1 = 1 - alpha
        self.Pey = alpha_1 * self.Pey + alpha * Pey
        self.Pyy = alpha_1 * self.Pyy + alpha * Pyy
        if self.Pyy < 1:
            self.Pyy = 1
        if self.Pey < MIN_LEAK * self.Pyy:
            self.Pey = MIN_LEAK * self.Pyy
        if self.Pey > self.Pyy:
            self.Pey = self.Pyy
        self.leak_estimate = self.Pey / self.Pyy
        if self.leak_estimate > 16383:
            self.leak_estimate = 32767
        RER = (.0001 * Sxx + 3. * self.leak_estimate * Syy) / See
        if RER < Sey * Sey / (1 + See * Syy):
            RER = Sey * Sey / (1 + See * Syy)
        if RER > .5:
            RER = .5
        # if (not self.adapted and self.sum_adapt > M and self.leak_estimate*Syy > .03*Syy):
        #     self.adapted = 1
        #
        # if self.adapted:
        #     for i in range(self.frame_size+1):
        #         r = self.leak_estimate*self.Yf[i]
        #         e = self.Rf[i]+1
        #         if r > .5*e:
        #             r = .5*e
        #         r = 0.7*r + 0.3*(RER*e)
        #         self.power_1[i] = (r/(e*self.power[i]+10))
        # else:
        adapt_rate = 0
        if Sxx > N * 1000:
            tmp32 = 0.25 * Sxx
            if tmp32 > .25 * See:
                tmp32 = .25 * See
            adapt_rate = tmp32 / See
        self.power_1 = adapt_rate / (self.power + 10)
        self.sum_adapt = self.sum_adapt + adapt_rate

        self.last_y[:self.frame_size] = self.last_y[self.frame_size:N]
        if self.adapted:
            self.last_y[self.frame_size:N] = mic - out
        return out

    # spx_r2_word32_t *ps;         /**< Current power spectrum */
    def speex_r2_preprocess_run(self, out):
        N = self.ps_size
        N3 = 2*N - self.frame_size
        N4 = self.frame_size - N3
        self.N3 = N3
        self.N4 = N4
        self.window2 = self.init_window2(self.window2, N3*2)
        self.nb_adapt += 1
        if self.nb_adapt > 20000:
            self.nb_adapt = 20000
        self.min_count += 1

        beta = max(.03, 1/self.nb_adapt)
        beta_1 = 1 - beta
        M = self.nbands
        if self.echo_state:
            self.residual_echo = self.r2_echo_get_residual(mic_idx=0)
            if not (self.residual_echo[0] >= 0 and self.residual_echo[0] < N*1e9):
                self.residual_echo = np.zeros((self.frame_size, 1),)
            for i in range(N):
                self.echo_noise[i] = max(0.6 * self.echo_noise[i], self.residual_echo[i])
            self.echo_noise[N:] = self.filterbank_r2_compute_bank(self.echo_noise)
        else:
            self.echo_noise = np.zeros((self.frame_size+self.nbands, 1), )
        self.preprocess_analysis(out)
        self.update_noise_prob()
        # Update the noise estimate for the frequencies where it can be
        for i in range(N):
            if not self.update_prob[i] or self.ps[i] < self.noise[i]:
                self.noise[i] = max(0, beta_1*self.noise[i]) + beta * self.ps[i]

        self.noise[N:] = self.filterbank_r2_compute_bank(self.noise).squeeze(1)
        # Special case for first frame
        if self.nb_adapt == 1:
            for i in range(N+M):
                self.old_ps[i] = self.ps[i] # ps这里可能需要改成局部变量，如果改成多通道
        if self.do_dereverb:
            # Todo:完成去混响
            return 0
        # Compute a posteriori SNR
        for i in range(N+M):
            tot_noise = 1+self.noise[i] + self.echo_noise[i, 0] + self.reverb_estimation[i]
            # A posteriori SNR = ps/noise - 1
            self.post[i] = self.ps[i] / tot_noise - 1
            self.post[i] = min(self.post[i], 100)
            self.post[i] = max(self.post[i], 0.00001)
            # Computing update gamma = .1 + .9*(old/(old+noise))^2
            gamma = .1 + .89*(self.old_ps[i]/(self.old_ps[i]+tot_noise))**2
            # A priori SNR update = gamma*max(0,post) + (1-gamma)*old/noise
            self.prior[i] = gamma*max(0, self.post[i]) + (1-gamma)*self.old_ps[i]/tot_noise
            self.prior[i] = min(self.prior[i], 100)
            self.prior[i] = max(self.prior[i], 0.00001)

        # Recursive average of the a priori SNR. A bit smoothed for the psd components
        self.zeta[0] = .7 * self.zeta[0] + .3 * self.prior[0]
        for i in range(1, N-1, 1):
            self.zeta[i] = .7 * self.zeta[i] + .15 * self.prior[i] + .075 * self.prior[i-1] + .075 * self.prior[i+1]

        for i in range(N-1, N+M, 1):
            self.zeta[i] = .7 * self.zeta[i] + .3 * self.prior[i]

        # Speech probability of presence for the entire frame is based on the average filterbank a priori SNR
        Zframe = 0
        for i in range(N, N+M, 1):
            Zframe += self.zeta[i]
        Pframe = .1 + .899 * 1/(1 +.15/(Zframe / self.nbands))
        effective_echo_suppress = (1 - Pframe)*self.echo_suppress + Pframe*self.echo_suppress_active
        self.gain_floor[N:] = self.comput_gain_floor(-5, effective_echo_suppress, self.noise[N:], self.echo_noise[N:], M)

        # Compute Ephraim & Malah gain speech probability of presence for each critical band (Bark scale)
        #      Technically this is actually wrong because the EM gaim assumes a slightly different probability
        #      distribution
        for i in range(N, N+M, 1):
            prior_ratio = self.prior[i]/(self.prior[i]+1)
            theta = prior_ratio * (1 + self.post[i])
            MM = self.hypergeom_gain(theta)
            # Gain with bound
            self.gain[i] = min(1, prior_ratio*MM)
            self.old_ps[i] = .2*self.old_ps[i] + .8*self.gain[i]**2 * self.ps[i]
            P1 = .199 + .8 * 1/(1+.15/(self.zeta[i]))
            q = 1 - Pframe * P1
            self.gain2[i] = 1/(1+(q/(1-q))*(1+self.prior[i])*np.exp(-theta))

        # Convert the EM gains and speech prob to linear frequency
        self.gain2[:N] = self.filterbank_r2_compute_psd16(self.gain2[N:])
        self.gain[:N] = self.filterbank_r2_compute_psd16(self.gain[N:])

        # Use 1 for linear gain resolution (best) or 0 for Bark gain resolution (faster)
        self.gain_floor[:N] = self.filterbank_r2_compute_psd16(self.gain_floor[N:])
        # Compute gain according to the Ephraim-Malah algorithm -- linear frequency
        for i in range(N):
            # Wiener filter gain
            prior_ratio = self.prior[i]/(self.prior[i]+1)
            theta = prior_ratio * (1 + self.post[i])
            MM = self.hypergeom_gain(theta)
            g = min(1, prior_ratio*MM)
            p = self.gain2[i]
            # Constrain the gain to be close to the Bark scale gain
            if .333 * g > self.gain[i]:
                g = 3 * self.gain[i]
            self.gain[i] = g
            # Save old power spectrum
            self.old_ps[i] = .2 * self.old_ps[i] + .8 * self.gain[i] ** 2 * self.ps[i]
            # Apply gain floor
            if self.gain[i] < self.gain_floor[i]:
                self.gain[i] = self.gain_floor[i]
            tmp = p * np.sqrt(self.gain[i]) + (1-p)*np.sqrt(self.gain_floor[i])
            self.gain2[i] = tmp**2

        sn_avg = 0
        self.cn = np.zeros((2*N, 1), )

        # noise suppression
        if not self.denoise_enable:
            for i in range(N+M):
                self.gain2[i] = 1
        # Apply computed gain
        for i in range(1, N, 1):
            self.ft[i] = self.ft[i].real * self.gain2[i] + (self.ft[i].imag * self.gain2[i])*1j
        self.ft[0] = self.gain2[0] * self.ft[0]
        self.ft[N] = self.gain2[N-1]*self.ft[N]
        self.frame = np.fft.irfft(self.ft, axis=0)*512
        # Scale back to original (lower) amplitude
        # Synthesis window (for WOLA)
        for i in range(2*self.frame_size):
            self.frame[i] = self.frame[i]*self.window2[i]
        # Synthesis window (for WOLA)
        for i in range(N3):
            out[i] = self.outbuf[i] + self.frame[i]

        out[N3:N3+N4] = self.frame[N3:N3+N4, 0]
        self.outbuf[:N3, 0] = self.frame[self.frame_size:self.frame_size+N3, 0]

        return out

    def preprocess_analysis(self, x):
        N = self.ps_size
        N3 = 2*N - self.frame_size
        N4 = self.frame_size - N3
        for i in range(N3):
            self.frame[i] = self.inbuf[i]
        for i in range(self.frame_size):
            self.frame[N3+i] = x[i]
        for i in range(N3):
            self.inbuf[i] = x[N4+i]
        for i in range(2*N):
            self.frame[i] = self.frame[i] * self.window2[i]

        self.ft = np.fft.rfft(self.frame, axis=0)/(2*N)
        self.ps[0] = self.ft[0] * self.ft[0]
        for i in range(1, N, 1):
            self.ps[i] = np.abs(self.ft[i])**2
        self.ps[self.frame_size:] = self.filterbank_r2_compute_bank(self.ps)

    def update_noise_prob(self):
        N = self.ps_size
        for i in range(1, N-1, 1):
            self.S[i] = .8 * self.S[i] + .05 * self.ps[i-1] + .1 * self.ps[i] + .05 * self.ps[i+1]
        self.S[0] = .8 * self.S[0] + .2 * self.ps[0]
        self.S[N-1] = .8 * self.S[N-1] + .2 * self.ps[N-1]

        if self.nb_adapt == 1:
            self.Smin = np.zeros((N, 1), )
            self.Stmp = np.zeros((N, 1), )

        if self.nb_adapt < 100:
            min_range = 15
        elif self.nb_adapt < 1000:
            min_range = 30
        elif self.nb_adapt < 10000:
            min_range = 30
        else:
            min_range = 30
        if self.min_count > min_range:
            self.min_count = 0
            for i in range(N):
                self.Smin[i] = min(self.Stmp[i], self.S[i])
                self.Stmp[i] = self.S[i]
        else:
            for i in range(N):
                self.Smin[i] = min(self.Smin[i], self.S[i])
                self.Stmp[i] = min(self.Stmp[i], self.S[i])

        for i in range(N):
            if .4*self.S[i] > self.Smin[i]:
                self.update_prob[i] = 1
            else:
                self.update_prob[i] = 0

    def init_window2(self, window2, len_1):
        for i in range(len_1):
            x = (4*i)/len_1
            inv = 0
            if x < 1:
                inv = inv
            elif x < 2:
                inv = 1
                x = x - 2
            elif x < 3:
                x = 2 -x
                inv = 1
            else:
                x = 2 - x + 2
            x = 1.271903 * x
            tmp = (0.5 - 0.5 * np.cos(0.5*np.pi*x))**2
            if inv:
                tmp = 1 - tmp
            window2[i] = np.sqrt(tmp)
        for i in range(len_1, self.ps_size*2, 1):
            window2[i] = 1
        if self.N4 > 0:
            for i in range(self.N3-1, -1, -1):
                self.window2[i+self.N3+self.N4] = self.window2[i+self.N3]
                self.window2[i+self.N3] = 1

        return window2

    def comput_gain_floor(self, noise_suppress, effective_echo_suppress, noise,  echo, len):
        noise_floor = np.exp(.2302585*noise_suppress)
        echo_floor = np.exp(.2302585*int(effective_echo_suppress))

        # Compute the gain floor based on different floors for the background noise and residual echo
        gain_floor = np.zeros((len, 1),)
        for i in range(len):
            gain_floor[i] = np.sqrt(noise_floor*noise[i] + echo_floor * echo[i])/np.sqrt(1+noise[i]+echo[i])
        return gain_floor

    def hypergeom_gain(self, xx):
        table = np.array([0.82157, 1.02017, 1.20461, 1.37534, 1.53363, 1.68092, 1.81865,
        1.94811, 2.07038, 2.18638, 2.29688, 2.40255, 2.50391, 2.60144,
        2.69551, 2.78647, 2.87458, 2.96015, 3.04333, 3.12431, 3.20326])
        x = xx
        integer = np.floor(2*x)
        ind = int(integer)
        if ind < 0:
            return 1
        if ind > 19:
            return 1 + .1296/x
        frac = 2*x - integer
        return ((1-frac)*table[ind] + frac*table[ind+1])/np.sqrt(x+.0001)

    def filterbank_r2_compute_bank(self, ps):
        mel = np.zeros((self.nbands, 1),)
        for i in range(self.frame_size):
            id = self.bank_left[i]
            mel[int(id)] += self.filter_left[i] * ps[i]
            id = self.bank_right[i]
            mel[int(id)] += self.filter_right[i] * ps[i]

        return mel

    def filterbank_r2_compute_psd16(self, mel):
        ps = np.zeros((self.frame_size, 1), )
        for i in range(self.frame_size):
            id1 = int(self.bank_left[i])
            id2 = int(self.bank_right[i])
            tmp = mel[id1] * self.filter_left[i]
            tmp += mel[id2] * self.filter_right[i]
            ps[i] = tmp
        return ps
    def r2_echo_get_residual(self, mic_idx=0):
        N = self.window_size
        for i in range(N):
            self.y[i] = self.window[i] * self.last_y[mic_idx*2*N+i]
        self.Y = np.fft.rfft(self.y, axis=0)/N
        residual_echo = np.zeros((self.frame_size + 1, 1), )
        residual_echo[1:, 0] = np.abs(self.Y[1:, 0])**2
        if self.leak_estimate > .5:
            leak2 = 1
        else:
            leak2 = 2 * self.leak_estimate
        for i in range(self.frame_size+1):
            residual_echo[i] = leak2 * residual_echo[i]

        return residual_echo
    def filter_dc_notch16(self, mic, mem):
        out = np.zeros_like(mic)
        den2 = self.notch_radius ** 2 + 0.7 * \
               (1 - self.notch_radius) * (1 - self.notch_radius)
        # out = signal.lfilter([self.notch_radius, self.notch_radius * -2, self.notch_radius],
        #                      [1, -2 * self.notch_radius, den2], mic, axis=0)
        for i in range(self.frame_size):
            vin = mic[i]
            vout = mem[0] + vin
            mem[0] = mem[1] + 2 * (-vin + self.notch_radius * vout)
            mem[1] = vin - (den2 * vout)
            out[i] = self.notch_radius * vout
        return out, mem

    def mdf_adjust_prop(self, ):
        N = self.window_size
        M = self.M
        C = self.C
        K = self.K
        prop = np.zeros((M, 1), )
        for i in range(M):
            tmp = 1
            for chan in range(C):
                for speak in range(K):
                    tmp = tmp + np.sum(np.abs(self.W[:N // 2 + 1, speak, i, chan]) ** 2)
            prop[i] = np.sqrt(tmp)
        max_sum = np.maximum(prop, 1)
        prop = prop + .1 * max_sum
        prop_sum = 1 + np.sum(prop)
        prop = 0.99 * prop / prop_sum
        return prop


if __name__ == "__main__":
    import soundfile as sf
    import librosa

    index = "nhtm_1"
    #data = wavfile.read(f"../midput/{index}_mic_out1.wav")
    mic, sr = sf.read(f"../midput/{index}_mic_out1.wav")
    ref, sr = sf.read(f"../midput/{index}_ref_out.wav")
    min_len = min(len(mic), len(ref))
    mic = mic[:16000*30]
    ref = ref[:16000*30]
    # 64 2048 for 8kHz.
    processor = MDF(sr, 256, 3072)
    e, y = processor.main_loop(ref, mic)

    sf.write(f'./output_1/{index}_traditional3_py2.wav', e, sr)
    # sf.write('y.wav', y, sr)
