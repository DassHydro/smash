! external library updated to work with Tapenade and wrapped

module mwd_bayesian_tools

    implicit none

    !integer,parameter::mrk= selected_real_kind(p=8) ! sp in SMASH vs. mrk in DMSL?
    integer, parameter::mrk = 8 ! sp in SMASH vs. mrk in DMSL?
    integer, parameter::mik = 8 !
    integer(mik), parameter::len_longStr = 256
    real(mrk), parameter::undefRN = -999999999._mrk    ! flag for undefined real numbers
    integer(mik), parameter::undefIN = -999999999            ! flag for undefined integer numbers
    real(mrk), parameter::mv = -0._mrk    ! missing value threshold -9999._mrk in BaM, 0 in Smash
    real(mrk), parameter::pi = 3.1415926535897932384626433832795028841971693993751_mrk

    ! BEN 2DO: try linking wit existing type in BMSL's BayesianEstimation_tools
    type, public:: PriorType
        character(250)::dist = 'FlatPrior' !$F90W char
        real(mrk), allocatable::par(:)
    end type PriorType

    ! type, public:: PriorListType
    !     type(PriorType),allocatable::prior(:)
    ! end type PriorListType

    public :: compute_logPost, compute_logLkh, compute_logPrior, compute_logH, MuFunk_vect, SigmaFunk_vect

contains

    subroutine PriorType_initialise(this, n)
        type(PriorType), intent(inout)::this
        integer, intent(in) :: n
        allocate (this%par(n))
    end subroutine PriorType_initialise

    ! subroutine PriorListType_initialise(this, n)
    !     type(PriorListType),intent(inout)::this
    !     integer, intent(in) :: n

    !     integer :: i

    !     allocate(this%prior(n))

    !     do i = 1, n
    !         call PriorType_initialise(this%prior(i), 0)
    !     end do

    ! end subroutine PriorListType_initialise

    subroutine compute_logLkh(obs, uobs, sim, mu_funk, mu_gamma, sigma_funk, sigma_gamma, logLkh, feas, isnull)

        real(mrk), intent(in)::obs(:, :), sim(:, :), uobs(:, :) ! nT*nS
        character(*), intent(in)::mu_funk, sigma_funk
        real(mrk), intent(in)::mu_gamma(:, :), sigma_gamma(:, :) ! nHyper*nS
        real(mrk), intent(out)::logLkh
        logical, intent(out)::feas, isnull
        ! locals
        character(len_longStr), parameter::procname = 'compute_logLkh'
        integer(mik) :: nS, nT, s, t, err
        real(mrk)::v, mu, sigma
        character(len_longStr)::mess

        ! Initialize
        logLkh = 0._mrk; feas = .true.; isnull = .false.

        ! Compute
        nT = size(obs, dim=1); nS = size(obs, dim=2)

        do s = 1, nS
            do t = 1, nT
                if (obs(t, s) >= mv .and. uobs(t, s) >= mv) then
                    call MuFunk_Apply(funk=mu_funk, par=mu_gamma(:, s), Y=sim(t, s), res=mu, err=err, mess=mess)
                    If (err > 0) then
                        mess = trim(procname)//': '//trim(mess)
                        feas = .false.
                        ! return removed for Tapenade
                    end if
                    call SigmaFunk_Apply(funk=sigma_funk, par=sigma_gamma(:, s), Y=sim(t, s), res=sigma, err=err, mess=mess)
                    If (err > 0) then
                        mess = trim(procname)//': '//trim(mess)
                        feas = .false.
                        ! return removed for Tapenade
                    end if
                    v = sigma**2 + Uobs(t, s)**2
                    if (v <= 0._mrk) then
                        feas = .false.
                        ! return removed for Tapenade
                        !exit
                    end if
                    logLkh = logLkh - 0.5_mrk*(log(2._mrk*pi) + log(v) + &
                                         & (obs(t, s) - sim(t, s) - mu)**2/v)
                end if
            end do
        end do

    end subroutine compute_logLkh

    subroutine compute_logPrior(theta, theta_prior, mu_gamma, mu_gamma_prior, sigma_gamma, sigma_gamma_prior, &
    & logPrior, feas, isnull)

        real(mrk), intent(in)::theta(:), mu_gamma(:, :), sigma_gamma(:, :)
        type(PriorType), intent(in)::theta_prior(:), mu_gamma_prior(:, :), sigma_gamma_prior(:, :)
        real(mrk), intent(out)::logPrior
        logical, intent(out)::feas, isnull
        ! locals
        real(mrk)::pdf
        real(mrk)::dummyTheta(size(theta), 1)
        type(PriorType)::dummyTheta_prior(size(theta_prior), 1)

        ! Initialize
        logPrior = 0.0_mrk; feas = .true.; isnull = .false.

        ! theta
        dummyTheta(:, 1) = theta
        dummyTheta_prior(:, 1) = theta_prior
        call compute_logPrior_engine(dummyTheta, dummyTheta_prior, pdf, feas, isnull)
        if ((.not. feas) .or. (isnull)) return
        logPrior = logPrior + pdf

        ! mu hyperparameters
        call compute_logPrior_engine(mu_gamma, mu_gamma_prior, pdf, feas, isnull)
        if ((.not. feas) .or. (isnull)) return
        logPrior = logPrior + pdf

        ! sigma hyperparameters
        call compute_logPrior_engine(sigma_gamma, sigma_gamma_prior, pdf, feas, isnull)
        if ((.not. feas) .or. (isnull)) return
        logPrior = logPrior + pdf

    end subroutine compute_logPrior

    subroutine compute_logH(logH, feas, isnull)

        real(mrk), intent(out)::logH
        logical, intent(out)::feas, isnull

        ! Initialize
        logH = 0._mrk; feas = .true.; isnull = .false.

    end subroutine compute_logH

    subroutine compute_logPost(obs, uobs, sim, theta, theta_prior, mu_funk, mu_gamma, mu_gamma_prior, &
        & sigma_funk, sigma_gamma, sigma_gamma_prior, logPost, logPrior, logLkh, logH, feas, isnull)

        real(mrk), intent(in)::obs(:, :), sim(:, :), uobs(:, :)
        character(*), intent(in)::mu_funk, sigma_funk
        real(mrk), intent(in)::theta(:), mu_gamma(:, :), sigma_gamma(:, :)
        type(PriorType), intent(in)::theta_prior(:), mu_gamma_prior(:, :), sigma_gamma_prior(:, :)
        real(mrk), intent(out)::logPost, logPrior, logLkh, logH
        logical, intent(out)::feas, isnull

        ! Initialize
        logPost = undefRN; feas = .true.; isnull = .false.

        ! Prior
        call compute_logPrior(theta, theta_prior, mu_gamma, mu_gamma_prior, sigma_gamma, sigma_gamma_prior, logPrior, feas, isnull)
        if ((.not. feas) .or. isnull) return
        ! Likelihood
        call compute_logLkh(obs, uobs, sim, mu_funk, mu_gamma, sigma_funk, sigma_gamma, logLkh, feas, isnull)
        if ((.not. feas) .or. isnull) return
        ! Hierarchical term
        call compute_logH(logH, feas, isnull)
        if ((.not. feas) .or. isnull) return
        ! Posterior
        logPost = logLkh + logPrior + logH

    end subroutine compute_logPost

    subroutine compute_logPrior_engine(x, x_prior, logPrior, feas, isnull)

        real(mrk), intent(in)::x(:, :)
        type(PriorType), intent(in)::x_prior(:, :)
        real(mrk), intent(out)::logPrior
        logical, intent(out)::feas, isnull
        ! locals
        integer(mik)::i, j, err
        real(mrk)::pdf
        character(250)::mess

        ! Initialize
        logPrior = 0.0_mrk; feas = .true.; isnull = .false.

        ! No prior <=> flat prior
        if (size(x_prior) == 0) return

        ! Add up individual log-priors
        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                call GetPdf(x_prior(i, j)%dist, x(i, j), x_prior(i, j)%par, .true., pdf, feas, isnull, err, mess)
                If (err > 0) feas = .false.
                ! return removed for Tapenade
                logPrior = logPrior + pdf
            end do
        end do

    end subroutine compute_logPrior_engine

    ! BEN 2DO: try replacing what's below with existing subs in BMSL's Distribution_tools
    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pure subroutine GetParNumber(DistID, npar, err, mess)

        !^**********************************************************************
        !^* Purpose: returns the number of parameters of a distribution
        !^**********************************************************************
        !^* Programmer: Ben Renard, University of Newcastle
        !^**********************************************************************
        !^* Last modified:30/05/2008
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.DistID, the distribution ID
        !^* OUT
        !^*    1.nPar, the numbr of parameters
        !^*    2.err, error code
        !^*    3.mess, error message
        !^**********************************************************************

        character(*), intent(in)::DistID
        integer(mik), intent(out)::nPar
        integer(mik), intent(out)::err
        character(*), intent(out)::mess

        err = 0; mess = ''; nPar = UndefIN
        select case (DistID)
        case ('FlatPrior')
            npar = 0
        case ('Gaussian', 'Uniform', 'LogNormal', 'Exponential')
            npar = 2
        case ('Triangle')
            npar = 3
        case default
            err = 1; mess = 'GetParNumber:Fatal:Unavailable Dist'
        end select

    end subroutine GetParNumber

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pure subroutine GetParName(DistID, name, err, mess)

        !^**********************************************************************
        !^* Purpose: returns the names of parameters of a distribution
        !^**********************************************************************
        !^* Programmer: Ben Renard, University of Newcastle
        !^**********************************************************************
        !^* Last modified:30/05/2008
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.DistID, the distribution ID
        !^* OUT
        !^*    1.name, parameters names
        !^*    2.err, error code
        !^*    3.mess, error message
        !^**********************************************************************

        character(*), intent(in)::DistID
        character(*), intent(out)::name(:)
        integer(mik), intent(out)::err
        character(*), intent(out)::mess
        !locals
        integer(mik)::npar

        err = 0; mess = ''; name = ''

        ! check size
        call GetParNumber(DistID, npar, err, mess)
        if (err > 0) then
            mess = 'GetParName: '//trim(mess); return
        end if
        if (size(name) /= npar) then
            err = 2; mess = 'GetParName: dimension mismatch'; return
        end if

        select case (DistID)
        case ('FlatPrior')
            ! no parameter
        case ('Gaussian')
            name(1) = 'mean'
            name(2) = 'standard_deviation'
        case ('LogNormal')
            name(1) = 'mean_log'
            name(2) = 'standard_deviation_log'
        case ('Exponential')
            name(1) = 'threshold'
            name(2) = 'scale'
        case ('Uniform')
            name(1) = 'lower_bound'
            name(2) = 'higher_bound'
        case ('Triangle')
            name(1) = 'peak'
            name(2) = 'lower_bound'
            name(3) = 'higher_bound'
        case default
            err = 1; mess = 'GetParName:Fatal:Unavailable Dist'
        end select

    end subroutine GetParName

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pure subroutine CheckParSize(DistID, par, ok, err, mess)

        !^**********************************************************************
        !^* Purpose: check size(par)=ParNumber(dist)
        !^**********************************************************************
        !^* Programmer: Ben Renard, University of Newcastle
        !^**********************************************************************
        !^* Last modified:30/05/2008
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.DistID, the distribution
        !^*    1.par, parameter vector
        !^* OUT
        !^*    1.ok
        !^*    2.err, error code
        !^*    3.mess, error message
        !^**********************************************************************

        character(*), intent(in)::DistID
        real(mrk), intent(in)::par(:)
        logical, intent(out)::ok
        integer(mik), intent(out)::err
        character(*), intent(out)::mess
        !locals
        integer(mik)::npar

        err = 0; mess = ''; ok = .false.
        call GetParNumber(DistID, npar, err, mess)
        if (err > 0) then
            mess = 'CheckParSize: '//trim(mess); return
        end if
        if (size(par) == npar) ok = .true.

    end subroutine CheckParSize

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pure subroutine GetParFeas(DistID, par, feas, err, mess)

        !^**********************************************************************
        !^* Purpose: check parameters feasability
        !^**********************************************************************
        !^* Programmer: Ben Renard, University of Newcastle
        !^**********************************************************************
        !^* Last modified:30/05/2008
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.DistID, the distribution ID
        !^*    2.par, parameter vector
        !^* OUT
        !^*    1.feas, feasability
        !^*    2.err, error code
        !^*    3.mess, error message
        !^**********************************************************************

        character(*), intent(in)::DistID
        real(mrk), intent(in)::par(:)
        logical, intent(out)::feas
        integer(mik), intent(out)::err
        character(*), intent(out)::mess
        ! Locals
        logical::ok

        err = 0; mess = ''; feas = .true.

        ! check size
        call CheckParSize(DistID, par, ok, err, mess)
        if (.not. ok) then
            err = 2; mess = 'GetParFeas: dimension mismatch'; return
        end if

        select case (DistID)
        case ('FlatPrior')
            ! Can't get it wrong!
        case ('Gaussian', 'LogNormal', 'Exponential')
            if (par(2) <= 0.0_mrk) feas = .false.
        case ('Uniform')
            if (par(2) <= par(1)) feas = .false.
        case ('Triangle')
            if (par(3) <= par(2) .or. par(1) <= par(2) .or. par(1) >= par(3)) feas = .false.
        case default
            err = 1; mess = 'GetParFeas:Fatal:Unavailable Dist'
        end select

    end subroutine GetParFeas

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    pure subroutine GetPdf(DistId, x, par, loga, pdf, feas, isnull, err, mess)

        !^**********************************************************************
        !^* Purpose: compute pdf(x|par) for DistID
        !^**********************************************************************
        !^* Programmer: Ben Renard, University of Newcastle
        !^**********************************************************************
        !^* Last modified:30/05/2008
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.DistID, the distribution
        !^*    2.x, value
        !^*    3.par, parameters
        !^*    4.loga, log-pdf or natural pdf?
        !^* OUT
        !^*    1.pdf, result
        !^*    2.feas, is par feasible?
        !^*    3.isnull, pdf==0? (usefull for log-pdf of bounded-support distribution)
        !^*    4.err, error code
        !^*    5.mess, error message
        !^**********************************************************************

        character(*), intent(in)::DistID
        real(mrk), intent(in)::x, par(:)
        logical, intent(in)::loga
        real(mrk), intent(out)::pdf
        logical, intent(out)::feas, isnull
        integer(mik), intent(out)::err
        character(*), intent(out)::mess
        !Locals

        !Init
        pdf = UndefRN; feas = .true.; isnull = .false.; err = 0; mess = ''

        !Feasability
        call GetParFeas(DistID, par, feas, err, mess)
        if (err > 0) then
            mess = 'GetPdf: '//trim(mess); return
        end if
        if (.not. feas) return

        !Compute
        select case (DistID)
        case ('FlatPrior')
            pdf = 0.0_mrk
        case ('Gaussian')
            pdf = -0.5_mrk*log(2._mrk*pi) - log(par(2)) - 0.5_mrk*((x - par(1))/par(2))**2
        case ('LogNormal')
            if (x <= 0.0_mrk) then
                isnull = .true.
            else
                pdf = -0.5_mrk*log(2._mrk*pi) - log(x*par(2)) - 0.5_mrk*((log(x) - par(1))/par(2))**2
            end if
        case ('Exponential')
            if (x < par(1)) then
                isnull = .true.
            else
                pdf = -1.0_mrk*log(par(2)) - (x - par(1))/par(2)
            end if
        case ('Uniform')
            if (x < par(1) .or. x > par(2)) then
                isnull = .true.
            else
                pdf = -1.0_mrk*log(par(2) - par(1))
            end if
        case ('Triangle')
            if (x < par(2) .or. x > par(3)) then
                isnull = .true.
            else
                if (x <= par(1)) then
                    pdf = log(2._mrk) + log(x - par(2)) - log(par(3) - par(2)) - log(par(1) - par(2))
                else
                    pdf = log(2._mrk) + log(par(3) - x) - log(par(3) - par(2)) - log(par(3) - par(1))
                end if
            end if
        case default
            err = 1; mess = 'GetPdf:Fatal:Unavailable Dist'
        end select

        ! Exponentiate if loga=.false.
        if (.not. loga) then
            if (isnull) then
                pdf = 0.0_mrk
            else
                pdf = exp(pdf)
            end if
        end if

    end subroutine GetPdf

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pure subroutine Sigmafunk_Apply(funk, par, Y, res, err, mess)
        !^**********************************************************************
        !^* Purpose: Apply the selected Sigmafunk
        !^**********************************************************************
        !^* Programmer: Ben Renard, Irstea Lyon
        !^**********************************************************************
        !^* Created: 29/04/2013, last modified: 05/08/2022, added 'Power'
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.funk, which function?? (e.g., 'Constant','Linear')
        !^*    2.par, parameters of funk
        !^*    3.Y, covariate of funk
        !^* OUT
        !^*    1.res, result
        !^*    2.err, error code; <0:Warning, ==0:OK, >0: Error
        !^*    3.mess, error message
        !^**********************************************************************
        character(*), intent(in)::funk
        real(mrk), intent(in)::par(:), Y
        real(mrk), intent(out)::res
        integer(mik), intent(out)::err
        character(*), intent(out)::mess
        ! locals
        character(250), parameter::procname = 'Sigmafunk_Apply'

        err = 0; mess = ''; res = undefRN

        select case (trim(funk))
        case ('Constant')
            res = par(1)
        case ('Linear')
            res = par(1) + par(2)*abs(Y)
        case ('Power')
            res = par(1) + par(2)*(abs(Y)**par(3))
        case ('Exponential')
            res = par(1) + (par(3) - par(1))*(1._mrk - exp(-(abs(Y)/par(2))**1))
        case ('Gaussian')
            res = par(1) + (par(3) - par(1))*(1._mrk - exp(-(abs(Y)/par(2))**2))
        case default
            err = 1; mess = trim(procname)//": unknown SigmaFunk"
        end select

    end subroutine SigmaFunk_Apply

    !%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pure subroutine MuFunk_Apply(funk, par, Y, res, err, mess)
        !^**********************************************************************
        !^* Purpose: Apply the selected MuFunk
        !^**********************************************************************
        !^* Programmer: Ben Renard, Irstea Lyon
        !^**********************************************************************
        !^* Created: 29/04/2013, last modified: 05/08/2022, added 'Power'
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.funk, which function?? (e.g., 'Constant','Linear')
        !^*    2.par, parameters of funk
        !^*    3.Y, covariate of funk
        !^* OUT
        !^*    1.res, result
        !^*    2.err, error code; <0:Warning, ==0:OK, >0: Error
        !^*    3.mess, error message
        !^**********************************************************************
        character(*), intent(in)::funk
        real(mrk), intent(in)::par(:), Y
        real(mrk), intent(out)::res
        integer(mik), intent(out)::err
        character(*), intent(out)::mess
        ! locals
        character(250), parameter::procname = 'MuFunk_Apply'

        err = 0; mess = ''; res = undefRN

        select case (trim(funk))
        case ('Zero')
            res = 0._mrk
        case ('Constant')
            res = par(1)
        case ('Linear')
            res = par(1) + par(2)*Y
        case default
            err = 1; mess = trim(procname)//": unknown MuFunk"
        end select

    end subroutine MuFunk_Apply

    pure subroutine SigmaFunk_vect(funk, par, Y, res)
        !^**********************************************************************
        !^* Purpose: Apply the selected Sigmafunk to an array Y(:,:)
        !^**********************************************************************
        !^* Programmer: Ben Renard & François Colleoni, INRAE Aix
        !^**********************************************************************
        !^* Created: 29/04/2013, last modified: 20/09/2023
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.funk, which function?? (e.g., 'Constant','Linear')
        !^*    2.par, parameters of funk
        !^*    3.Y, covariate of funk
        !^* OUT
        !^*    1.res, result
        !^*    2.err, error code; <0:Warning, ==0:OK, >0: Error
        !^*    3.mess, error message
        !^**********************************************************************
        character(*), intent(in)::funk
        real(mrk), intent(in)::par(:, :), Y(:, :)
        real(mrk), intent(out)::res(:, :)
        ! locals
        integer(mik)::i, j, err
        character(250)::mess

        res = undefRN

        do j = 1, size(Y, 2)
            do i = 1, size(Y, 1)
                call SigmaFunk_Apply(funk, par(:, j), Y(i, j), res(i, j), err, mess)
            end do
        end do

    end subroutine SigmaFunk_vect

    pure subroutine MuFunk_vect(funk, par, Y, res)
        !^**********************************************************************
        !^* Purpose: Apply the selected Mufunk to an array Y(:,:)
        !^**********************************************************************
        !^* Programmer: Ben Renard & François Colleoni, INRAE Aix
        !^**********************************************************************
        !^* Created: 29/04/2013, last modified: 20/09/2023
        !^**********************************************************************
        !^* Comments:
        !^**********************************************************************
        !^* References:
        !^**********************************************************************
        !^* 2Do List:
        !^**********************************************************************
        !^* IN
        !^*    1.funk, which function?? (e.g., 'Constant','Linear')
        !^*    2.par, parameters of funk
        !^*    3.Y, covariate of funk
        !^* OUT
        !^*    1.res, result
        !^*    2.err, error code; <0:Warning, ==0:OK, >0: Error
        !^*    3.mess, error message
        !^**********************************************************************
        character(*), intent(in)::funk
        real(mrk), intent(in)::par(:, :), Y(:, :)
        real(mrk), intent(out)::res(:, :)
        ! locals
        integer(mik)::i, j, err
        character(250)::mess

        res = undefRN

        do j = 1, size(Y, 2)
            do i = 1, size(Y, 1)
                call MuFunk_Apply(funk, par(:, j), Y(i, j), res(i, j), err, mess)
            end do
        end do

    end subroutine MuFunk_vect

end module mwd_bayesian_tools
