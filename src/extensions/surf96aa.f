! use tregn96 from cps to calculate the dcdL and dcdA
        include "surfdisp96.f"
        include "tregn96.f"

        subroutine depthkernel(thkm,vpm,vsm,rhom,nlayer,nrefine,
     &                         iflsph,iwave,mode,igr,nperiods,t,c,
     &                         Lsen_Gsc,dcR_dA,dcR_dL,err)
        implicit none

        integer i,j,k,ii
        integer nlayer,iflsph,iwave,mode,igr,nperiods,err
        
        integer NL
        parameter(NL=200)
        integer NP
        parameter (NP=60)
        integer nrefine,nlayerr
        real*4 thkm(NL),vpm(NL),vsm(NL),rhom(NL)
        real*4 thkr(NL),vpr(NL),vsr(NL),rhor(NL)
        double precision t(NP),c(NP)

        ! for tregn96
        real t_in(nperiods), cp_in(nperiods)
        real TA_in(NL), TC_in(NL), TF_in(NL)
        real TL_in(NL), TN_in(NL), TRho_in(NL)
        real qp(NL), qs(NL), etap(NL)
        real etas(NL), frefp(NL), frefs(NL)

        real*4 dcdah(NP,NL),dcdn(NP,NL)
        real*4 dcdbv(NP,NL)
        real*4 dcR_dL(nperiods,(nlayer-1)*nrefine)
        real*4 dcR_dA(nperiods,(nlayer-1)*nrefine)
        real*4 Lsen_Gsc(nperiods, (nlayer-1)*nrefine)

        Lsen_Gsc=0.0

        err=0
cf2py intent(out) err, Lsen_Gsc, dcR_dA, dcR_dL

        call surfdisp96(thkm, vpm, vsm, rhom, nlayer, iflsph, iwave, mode, igr,
     &       nperiods, t, c, err)

        !------------------------------------------------------------------!
        ! refine layers
        do i = 1, nlayer
            do j = 1, nrefine
                ii = (i-1)*nrefine+j
                thkr(ii) = thkm(i)/nrefine
                vpr(ii) = vpm(i)
                vsr(ii) = vsm(i)
                rhor(ii) = rhom(i)
                ! last layer is halfspace, no need to refine
                if (i == nlayer) exit
            enddo
        enddo

        nlayerr = ii

        do i = 1, nlayerr
            TA_in(i)=rhor(i)*vpr(i)**2
            TC_in(i)=TA_in(i)
            TL_in(i)=rhor(i)*vsr(i)**2
            TN_in(i)=TL_in(i)
            TF_in(i)=1.0*(TA_in(i) - 2 * TL_in(i))
            TRho_in(i)=rhor(i)
        enddo

        qp(1:nlayerr)=150.0
        qs(1:nlayerr)=50.0
        etap(1:nlayerr)=0.00
        etas(1:nlayerr)=0.00
        frefp(1:nlayerr)=1.00
        frefs(1:nlayerr)=1.00

        cp_in(1:nperiods)=sngl(c(1:nperiods))
        t_in(1:nperiods)=sngl(t(1:nperiods))


        ! ! write(6, *)'tregn96'
        call tregn96(nlayerr, thkr, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in,
     &           qp, qs, etap, etas, frefp, frefs,
     &           nperiods, t_in, cp_in(1:nperiods),
     &           dcdah, dcdbv, dcdn)

        do i=1,nperiods
            k=0
            do j=1,nlayerr-1
                k=k+1
                dcR_dA(i,j) = (0.5/(rhor(k)*vpr(k))*dcdah(i, k) - TF_in(k)/((TA_in(k)-2.0*TL_in(k))**2)*dcdn(i,k))!*thkr(j)
                dcR_dL(i,j) = (0.5/(rhor(k)*vsr(k))*dcdbv(i, k) + 2.0*TF_in(k)/((TA_in(k)-2.0*TL_in(k))**2)*dcdn(i,k))!*thkr(j)
                Lsen_Gsc(i,j)=dcR_dA(i,j)*TA_in(k)+dcR_dL(i,j)*TL_in(k)
            enddo
        enddo

            
        !
        ! ! write(*,*)"nsublay:", nsublay(1:nz)
        ! do i=1,kmaxRc  ! period
        !     k=0
        !     do j=1,nz-1                ! inversion layer
        !         do jjj=1,nsublay(j)    ! refined layer k-th in jth inversion layer
        !             k=k+1
        !             dcR_dA = 0.5/(rrho(k)*rvp(k))*dcdah(i, k) - TF_in(k)/((TA_in(k)-2.0*TL_in(k))**2)*dcdn(i,k)
        !             dcR_dL = 0.5/(rrho(k)*rvs(k))*dcdbv(i, k) + 2.0*TF_in(k)/((TA_in(k)-2.0*TL_in(k))**2)*dcdn(i,k)
        !             Lsen_Gsc(post,i,j)=Lsen_Gsc(post,i,j)+dcR_dA*TA_in(k)+dcR_dL*TL_in(k)
        !         enddo
        !     enddo
        ! enddo

        end
