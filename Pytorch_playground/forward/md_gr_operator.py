import numpy as np


class GR5Model:


    def __init__(self, setup, mesh, input_data, parameters):
        self.setup = setup
        self.mesh = mesh
        self.input_data = input_data
        self.ci = parameters['ci']
        self.cp = parameters['cp']
        self.ct = parameters['ct']
        self.kexc = parameters['kexc']
        self.aexc = parameters['aexc']
        self.llr = parameters['llr']
        self.qt = np.zeros((mesh.nrow, mesh.ncol, setup.ntime_step)) # hydrological module output
        self.q = np.zeros((mesh.nrow, mesh.ncol, setup.ntime_step)) # routing module output
        self.hi = np.zeros((mesh.nrow, mesh.ncol))
        self.hp = np.zeros((mesh.nrow, mesh.ncol))
        self.ht = np.zeros((mesh.nrow, mesh.ncol))

    def gr_interception(self, prcp, pet, ci, hi):

        ei = min(pet, prcp + hi*ci)
        pn = max(0, prcp - ci*(1 - hi) - ei)
        en = pet - ei
        hi = hi + (prcp - ei - pn)/ci
        return pn, en, hi

    def gr_production(self, pn, en, cp, beta, hp):

        inv_cp = 1/cp
        pr = 0
        ps = cp*(1 - hp*hp)*np.tanh(pn*inv_cp)/(1 + hp*np.tanh(pn*inv_cp))
        es = (hp*cp)*(2 - hp)*np.tanh(en*inv_cp)/(1 + (1 - hp)*np.tanh(en*inv_cp))
        hp_imd = hp + (ps - es)*inv_cp

        if pn > 0:
        
            pr = pn - (hp_imd - hp)*cp
        
        perc = (hp_imd*cp)*(1 - (1 + (hp_imd/beta)**4)**(-0.25))
        hp = hp_imd - perc*inv_cp

        return pr, perc, hp

    def gr_exchange(self, kexc, ht):
        l = kexc*ht**3.5
        return l

    def gr_threshold_exchange(self, kexc, aexc, ht):
        l = kexc*(ht - aexc)
        return l

    def gr_transfer(self, n, prcp, pr, ct, ht):
        nm1 = n - 1
        d1pnm1 = 1/nm1
        if prcp < 0:
            pr_imd = ((ht*ct)**(-nm1) - ct**(-nm1))**(-d1pnm1) - (ht*ct)
        else:
            pr_imd = pr
        ht_imd = max(1.e-6, ht + pr_imd/ct)
        ht = (((ht_imd*ct)**(-nm1) + ct**(-nm1))**(-d1pnm1))/ct
        q = (ht_imd - ht)*ct
        return q, ht
    
    def gr5_time_step(self, time_step):

        prcp = self.get_atmos_data_time_step(self.mesh, self.input_data, time_step, "prcp")
        pet = self.get_atmos_data_time_step(self.mesh, self.input_data, time_step, "pet")

        for col in range(self.mesh.ncol):
            for row in range(self.mesh.nrow):
                if self.mesh.active_cell[row, col] == 0 or self.mesh.local_active_cell[row, col] == 0:
                    continue

                if prcp[row, col] >= 0 and pet[row, col] >= 0:
                    pn, en = self.gr_interception(prcp[row, col], pet[row, col], self.ci[row, col], self.hi[row, col])
                    pr, perc = self.gr_production(pn, en, self.cp[row, col], 9/4, self.hp[row, col])
                    l = self.gr_threshold_exchange(self.kexc[row, col], self.aexc[row, col], self.ht[row, col])
                
                else:
                    pr = 0
                    perc = 0
                    l = 0

                prr = 0.9*(pr + perc) + l
                prd = 0.1*(pr + perc)

                qr, _ = self.gr_transfer(5, prcp[row, col], prr, self.ct[row, col], self.ht[row, col])
                qd = max(0, prd + l)

                self.qt[row, col, time_step] = qr + qd

                # Transform from mm/dt to m3/s
                self.qt[row, col, time_step] = self.qt[row, col, time_step]*1e-3*self.mesh.dx[row, col]*self.mesh.dy[row, col]/self.setup.dt

                self.q[row, col, time_step] = self.qt[row, col, time_step]

    
                qup = self.upstream_discharge(row, col, time_step)

                hlr, qup, q = self.linear_routing(row, col, qup, time_step)


        return self.hi, self.hp, self.ht, self.qt
    
    def upstream_discharge(self, row, col, time_step):
        drow = [1, 1, 0, -1, -1, -1, 0, 1]
        dcol = [0, -1, -1, -1, 0, 1, 1, 1]

        qup = 0.0

        for i in range(8):
            row_imd = row + drow[i]
            col_imd = col + dcol[i]

            if row_imd < 1 or row_imd > self.mesh.nrow or col_imd < 1 or col_imd > self.mesh.ncol:
                continue

            if self.mesh.flwdir[row_imd, col_imd] == i:
                qup += self.q[row_imd, col_imd, time_step]

        return qup

    def linear_routing(self, row, col, qup, time_step):

        llr = self.llr[row, col]
        dx = self.mesh.dx[row, col]
        dy = self.mesh.dy[row, col]
        dt = self.setup.dt
        flwacc = self.mesh.flwacc[row, col]
        qup = (qup * dt) / (0.001 * (flwacc - dx * dy))

        hlr_imd = self.hlr[row, col] + qup

        self.hlr[row, col] = hlr_imd * np.exp(-dt / (llr * 60))

        self.q[row, col, time_step] = self.q[row, col, time_step] + (hlr_imd - self.hlr[row, col]) * 0.001 * (flwacc - dx * dy) / dt

        return self.hlr[row, col], qup

    def get_atmos_data_time_step(self, mesh, input_data, time_step, key):
        vle = np.zeros((mesh.nrow, mesh.ncol))

        if key.strip() == "prcp":
            vle = input_data.atmos_data.prcp[:, :, time_step]
        elif key.strip() == "pet":
            vle = input_data.atmos_data.pet[:, :, time_step]
        elif key.strip() == "temp":
            vle = input_data.atmos_data.temp[:, :, time_step]

        return vle



    def simulation(self):

        for time_step in range(self.setup.ntime_step):
            self.gr5_time_step(time_step)

        return self.qt, self.q




