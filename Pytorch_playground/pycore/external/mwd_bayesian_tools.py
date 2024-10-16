import numpy as np
import math

# Constants
mrk = 8
mik = 8
len_longStr = 256
undefRN = -999999999.0
undefIN = -999999999
mv = 0.0
pi = 3.1415926535897932384626433832795028841971693993751

class PriorType:
    def __init__(self, n):
        self.dist = 'FlatPrior'
        self.par = np.zeros(n)

class BayesianTools:
    def __init__(self):
        pass

    def compute_logLkh(self, obs, uobs, sim, mu_funk, mu_gamma, sigma_funk, sigma_gamma):
        logLkh = 0.0
        feas = True
        isnull = False
        nT, nS = obs.shape
        for s in range(nS):
            for t in range(nT):
                if obs[t, s] >= mv and uobs[t, s] >= mv:
                    mu, err, mess = self.MuFunk_Apply(mu_funk, mu_gamma[:, s], sim[t, s])
                    if err > 0:
                        mess = 'compute_logLkh: ' + mess
                        feas = False
                    sigma, err, mess = self.SigmaFunk_Apply(sigma_funk, sigma_gamma[:, s], sim[t, s])
                    if err > 0:
                        mess = 'compute_logLkh: ' + mess
                        feas = False
                    v = sigma**2 + uobs[t, s]**2
                    if v <= 0.0:
                        feas = False
                    logLkh = logLkh - 0.5 * (math.log(2.0 * pi) + math.log(v) + (obs[t, s] - sim[t, s] - mu)**2 / v)
        return logLkh, feas, isnull
    
    def compute_logPrior(self, theta, theta_prior, mu_gamma, mu_gamma_prior, sigma_gamma, sigma_gamma_prior):
        logPrior = 0.0
        feas = True
        isnull = False
        dummyTheta = np.array([theta])
        dummyTheta_prior = np.array([theta_prior])
        pdf, feas, isnull = self.compute_logPrior_engine(dummyTheta, dummyTheta_prior)
        if not feas or isnull:
            return logPrior, feas, isnull
        logPrior += pdf
        pdf, feas, isnull = self.compute_logPrior_engine(mu_gamma, mu_gamma_prior)
        if not feas or isnull:
            return logPrior, feas, isnull
        logPrior += pdf
        pdf, feas, isnull = self.compute_logPrior_engine(sigma_gamma, sigma_gamma_prior)
        if not feas or isnull:
            return logPrior, feas, isnull
        logPrior += pdf
        return logPrior, feas, isnull

    def compute_logH(self):
        logH = 0.0
        feas = True
        isnull = False
        return logH, feas, isnull

    def compute_logPost(self, obs, uobs, sim, theta, theta_prior, mu_funk, mu_gamma, mu_gamma_prior, sigma_funk, sigma_gamma, sigma_gamma_prior):
        logPost = undefRN
        feas = True
        isnull = False
        logPrior, feas, isnull = self.compute_logPrior(theta, theta_prior, mu_gamma, mu_gamma_prior, sigma_gamma, sigma_gamma_prior)
        if not feas or isnull:
            return logPost, logPrior, None, None, feas, isnull
        logLkh, feas, isnull = self.compute_logLkh(obs, uobs, sim, mu_funk, mu_gamma, sigma_funk, sigma_gamma)
        if not feas or isnull:
            return logPost, logPrior, logLkh, None, feas, isnull
        logH, feas, isnull = self.compute_logH()
        if not feas or isnull:
            return logPost, logPrior, logLkh, logH, feas, isnull
        logPost = logLkh + logPrior + logH
        return logPost, logPrior, logLkh, logH, feas, isnull

    def compute_logPrior_engine(self, x, x_prior):
        logPrior = 0.0
        feas = True
        isnull = False
        if len(x_prior) == 0:
            return logPrior, feas, isnull
        for j in range(x.shape[1]):
            for i in range(x.shape[0]):
                pdf, feas, isnull, err, mess = self.GetPdf(x_prior[i, j].dist, x[i, j], x_prior[i, j].par, True)
                if err > 0:
                    feas = False
                logPrior += pdf
        return logPrior, feas, isnull
    

    def get_par_number(self, dist_id):
        err = 0
        mess = ''
        n_par = None
        if dist_id == 'FlatPrior':
            n_par = 0
        elif dist_id in ['Gaussian', 'Uniform', 'LogNormal', 'Exponential']:
            n_par = 2
        elif dist_id == 'Triangle':
            n_par = 3
        else:
            err = 1
            mess = 'GetParNumber:Fatal:Unavailable Dist'
        return n_par, err, mess

    def get_par_name(self, dist_id):
        err = 0
        mess = ''
        name = []
        n_par, err, mess = self.get_par_number(dist_id)
        if err > 0:
            mess = 'GetParName: ' + mess
            return name, err, mess
        if dist_id == 'Gaussian':
            name = ['mean', 'standard_deviation']
        elif dist_id == 'LogNormal':
            name = ['mean_log', 'standard_deviation_log']
        elif dist_id == 'Exponential':
            name = ['threshold', 'scale']
        elif dist_id == 'Uniform':
            name = ['lower_bound', 'higher_bound']
        elif dist_id == 'Triangle':
            name = ['peak', 'lower_bound', 'higher_bound']
        else:
            err = 1
            mess = 'GetParName:Fatal:Unavailable Dist'
        return name, err, mess

    def check_par_size(self, dist_id, par):
        ok = False
        err = 0
        mess = ''
        n_par, err, mess = self.get_par_number(dist_id)
        if err > 0:
            mess = 'CheckParSize: ' + mess
            return ok, err, mess
        if len(par) == n_par:
            ok = True
        return ok, err, mess

    def get_par_feas(self, dist_id, par):
        feas = True
        err = 0
        mess = ''
        ok, err, mess = self.check_par_size(dist_id, par)
        if not ok:
            err = 2
            mess = 'GetParFeas: dimension mismatch'
            return feas, err, mess
        if dist_id in ['Gaussian', 'LogNormal', 'Exponential']:
            if par[1] <= 0:
                feas = False
        elif dist_id == 'Uniform':
            if par[1] <= par[0]:
                feas = False
        elif dist_id == 'Triangle':
            if par[2] <= par[1] or par[0] <= par[1] or par[0] >= par[2]:
                feas = False
        else:
            err = 1
            mess = 'GetParFeas:Fatal:Unavailable Dist'
        return feas, err, mess
    
    def get_pdf(self, dist_id, x, par, loga):
        pdf = None
        feas = True
        isnull = False
        err = 0
        mess = ''

        feas, err, mess = self.get_par_feas(dist_id, par)
        if err > 0:
            mess = 'GetPdf: ' + mess
            return pdf, feas, isnull, err, mess
        if not feas:
            return pdf, feas, isnull, err, mess

        if dist_id == 'FlatPrior':
            pdf = 0.0
        elif dist_id == 'Gaussian':
            pdf = -0.5 * np.log(2.0 * np.pi) - np.log(par[1]) - 0.5 * ((x - par[0]) / par[1]) ** 2
        elif dist_id == 'LogNormal':
            if x <= 0.0:
                isnull = True
            else:
                pdf = -0.5 * np.log(2.0 * np.pi) - np.log(x * par[1]) - 0.5 * ((np.log(x) - par[0]) / par[1]) ** 2
        elif dist_id == 'Exponential':
            if x < par[0]:
                isnull = True
            else:
                pdf = -1.0 * np.log(par[1]) - (x - par[0]) / par[1]
        elif dist_id == 'Uniform':
            if x < par[0] or x > par[1]:
                isnull = True
            else:
                pdf = -1.0 * np.log(par[1] - par[0])
        elif dist_id == 'Triangle':
            if x < par[1] or x > par[2]:
                isnull = True
            else:
                if x <= par[0]:
                    pdf = np.log(2.0) + np.log(x - par[1]) - np.log(par[2] - par[1]) - np.log(par[0] - par[1])
                else:
                    pdf = np.log(2.0) + np.log(par[2] - x) - np.log(par[2] - par[1]) - np.log(par[2] - par[0])
        else:
            err = 1
            mess = 'GetPdf:Fatal:Unavailable Dist'

        if not loga:
            if isnull:
                pdf = 0.0
            else:
                pdf = np.exp(pdf)

        return pdf, feas, isnull, err, mess

    def sigmafunk_apply(self, funk, par, y):
        res = None
        err = 0
        mess = ''

        if funk == 'Constant':
            res = par[0]
        elif funk == 'Linear':
            res = par[0] + par[1] * abs(y)
        elif funk == 'Power':
            res = par[0] + par[1] * (abs(y) ** par[2])
        elif funk == 'Exponential':
            res = par[0] + (par[2] - par[0]) * (1.0 - np.exp(-(abs(y) / par[1]) ** 1))
        elif funk == 'Gaussian':
            res = par[0] + (par[2] - par[0]) * (1.0 - np.exp(-(abs(y) / par[1]) ** 2))
        else:
            err = 1
            mess = 'Sigmafunk_Apply: unknown SigmaFunk'

        return res, err, mess

    def mufunk_apply(self, funk, par, y):
        res = None
        err = 0
        mess = ''

        if funk == 'Zero':
            res = 0.0
        elif funk == 'Constant':
            res = par[0]
        elif funk == 'Linear':
            res = par[0] + par[1] * y
        else:
            err = 1
            mess = 'MuFunk_Apply: unknown MuFunk'

        return res, err, mess
    

    def sigmafunk_vect(self, funk, par, Y):
        res = np.full(Y.shape, np.nan)
        err = 0
        mess = ''

        for j in range(Y.shape[1]):
            for i in range(Y.shape[0]):
                res[i, j], err, mess = self.sigmafunk_apply(funk, par[:, j], Y[i, j])
                if err > 0:
                    break
            if err > 0:
                break

        return res, err, mess

    def mufunk_vect(self, funk, par, Y):
        res = np.full(Y.shape, np.nan)
        err = 0
        mess = ''

        for j in range(Y.shape[1]):
            for i in range(Y.shape[0]):
                res[i, j], err, mess = self.mufunk_apply(funk, par[:, j], Y[i, j])
                if err > 0:
                    break
            if err > 0:
                break

        return res, err, mess