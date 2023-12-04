
        subroutine init_dccurve(verbose)

        implicit none

        integer verbose

        call dispersion_curve_init(verbose)

        end subroutine


c       nLayers and nSamples are not used in the final Python function
        subroutine get_disp_curve(nLayers, nSamples, nModes, group,
     &                            h, vp, vs, rho, periods, 
     &                            slowness, ray)

        implicit none

        integer i
        integer nLayers
        integer nSamples
        integer nModes
        integer group
        integer ray
        real*8 h(nLayers)
        real*8 vp(nLayers)
        real*8 vs(nLayers)
        real*8 rho(nLayers)
        real*8 periods(nSamples)
        real*8 omega(nSamples)
        real*8 slowness(nSamples*nModes)

        !f2py intent(in) :: nLayers,nSamples,nModes,group,h,vp,vs,rho,periods
        !f2py intent(hide), depend(h) :: nLayers = shape(h, 0)
        !f2py intent(hide), depend(h) :: nSamples = shape(omega,0) 
        !f2py intent(out) :: slowness

        do i=1,nSamples
          omega(i)=2*3.141592*1./periods(i)
        end do

        do i=1,nSamples
            slowness(i) = 0.0
        end do

        if (omega(1)>omega(nSamples)) then
            omega = omega(nsamples:1:-1)
        endif


c       Currently, the Fortran interface does not return if computation
c       was successful or not. If not, all slowness values are null.
c       This feature may evolve in the future.
        if (ray==1) then
            call dispersion_curve_rayleigh(nLayers, h, vp, vs, rho,
     &                                 nSamples, omega,
     &                                 nModes, slowness, group)
        else
            call dispersion_curve_love(nLayers, h, vp, vs, rho,
     &                                 nSamples, omega,
     &                                 nModes, slowness, group)
        endif

c        print*,nLayers
c        print*,nSamples
c        print*,nModes
c        print*,group
c        print*,h
c        print*,vp,vs,rho,omega



        
        end subroutine
