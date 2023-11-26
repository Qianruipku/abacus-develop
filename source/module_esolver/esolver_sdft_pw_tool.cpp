#include <chrono>

#include "esolver_sdft_pw.h"
#include "module_base/complexmatrix.h"
#include "module_base/constants.h"
#include "module_base/global_function.h"
#include "module_base/global_variable.h"
#include "module_base/memory.h"
#include "module_base/timer.h"
#include "module_base/vector3.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"
#include "module_hsolver/hsolver_pw_sdft.h"

#define TWOSQRT2LN2 2.354820045030949 // FWHM = 2sqrt(2ln2) * \sigma
#define FACTOR 1.839939223835727e7
namespace ModuleESolver
{
struct parallel_distribution
{
    parallel_distribution(const int& num_all, const int& np, const int myrank)
    {
        int num_per = num_all / np;
        int st_per = num_per * myrank;
        int re = num_all % np;
        if (myrank < re)
        {
            ++num_per;
            st_per += myrank;
        }
        else
        {
            st_per += re;
        }
        this->start = st_per;
        this->num_per = num_per;
    }
    int start;
    int num_per;
};


void ESolver_SDFT_PW::check_che(const int nche_in)
{
    //------------------------------
    //      Convergence test
    //------------------------------
    bool change = false;
    const int nk = kv.nks;
    ModuleBase::Chebyshev<double> chetest(nche_in);
    Stochastic_Iter& stoiter = ((hsolver::HSolverPW_SDFT*)phsol)->stoiter;
    Stochastic_hchi& stohchi = stoiter.stohchi;
    int ntest0 = 5;
    stohchi.Emax = pw_wfc->gk_ecut * pw_wfc->tpiba2;
    if (GlobalV::NBANDS > 0)
    {
        double tmpemin = 1e10;
        for (int ik = 0; ik < nk; ++ik)
        {
            tmpemin = std::min(tmpemin, this->pelec->ekb(ik, GlobalV::NBANDS - 1));
        }
        stohchi.Emin = tmpemin;
    }
    else
    {
        stohchi.Emin = 0;
    }
    for (int ik = 0; ik < nk; ++ik)
    {
        this->p_hamilt->updateHk(ik);
        stoiter.stohchi.current_ik = ik;
        const int npw = kv.ngk[ik];
        std::complex<double>* pchi = nullptr;
        int ntest = std::min(ntest0, stowf.nchip[ik]);
        for (int i = 0; i < ntest; ++i)
        {
            if (INPUT.nbands_sto == 0)
            {
                pchi = new std::complex<double>[npw];
                for (int ig = 0; ig < npw; ++ig)
                {
                    double rr = std::rand() / double(RAND_MAX);
                    double arg = std::rand() / double(RAND_MAX);
                    pchi[ig] = std::complex<double>(rr * cos(arg), rr * sin(arg));
                }
            }
            else if (GlobalV::NBANDS > 0)
            {
                pchi = &stowf.chiortho[0](ik, i, 0);
            }
            else
            {
                pchi = &stowf.chi0[0](ik, i, 0);
            }
            while (1)
            {
                bool converge;
                converge = chetest.checkconverge(&stohchi,
                                                 &Stochastic_hchi::hchi_norm,
                                                 pchi,
                                                 npw,
                                                 stohchi.Emax,
                                                 stohchi.Emin,
                                                 2.0);

                if (!converge)
                {
                    change = true;
                }
                else
                {
                    break;
                }
            }
            if (INPUT.nbands_sto == 0)
            {
                delete[] pchi;
            }
        }

        if (ik == nk - 1)
        {
#ifdef __MPI
            MPI_Allreduce(MPI_IN_PLACE, &stohchi.Emax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &stohchi.Emin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
            stoiter.stofunc.Emax = stohchi.Emax;
            stoiter.stofunc.Emin = stohchi.Emin;
            GlobalV::ofs_running << "New Emax " << stohchi.Emax << " ; new Emin " << stohchi.Emin << std::endl;
            change = false;
        }
    }
}

int ESolver_SDFT_PW::set_cond_nche(const double dt, const int nbatch, const double cond_thr)
{
    int nche_guess = 1000;
    ModuleBase::Chebyshev<double> chemt(nche_guess);
    Stochastic_Iter& stoiter = ((hsolver::HSolverPW_SDFT*)phsol)->stoiter;
    const double mu = this->pelec->eferm.ef;
    stoiter.stofunc.mu = mu;
    stoiter.stofunc.t = dt * nbatch;
    chemt.calcoef_pair(&stoiter.stofunc, &Sto_Func<double>::ncos, &Sto_Func<double>::n_sin);

    int nche;
    bool find = false;
    std::ofstream cheofs("Chebycoef");
    for (int i = 1; i < nche_guess; ++i)
    {
        double error = std::abs(chemt.coef_complex[i] / chemt.coef_complex[0]);
        cheofs << std::setw(5) << i << std::setw(20) << error << std::endl;
        if (!find && error < cond_thr)
        {
            nche = i + 1;
            std::cout << "set N order of Chebyshev for KG as " << nche << std::endl;
            find = true;
        }
    }
    cheofs.close();

    if (!find)
    {
        ModuleBase::WARNING_QUIT("ESolver_SDFT_PW", "N order of Chebyshev for KG will be larger than 1000!");
    }

    return nche;
}

void ESolver_SDFT_PW::calj12(const int& ik,
                             const int& perbands,
                             psi::Psi<std::complex<double>>& chi,
                             psi::Psi<std::complex<double>>& j1chi,
                             psi::Psi<std::complex<double>>& j2chi,
                             psi::Psi<std::complex<double>>& tmpvchi,
                             psi::Psi<std::complex<double>>& tmphchi,
                             hamilt::Velocity& velop)
{
    const double mu = this->pelec->eferm.ef;
    Stochastic_Iter& stoiter = ((hsolver::HSolverPW_SDFT*)phsol)->stoiter;
    const int npw = kv.ngk[ik];

    psi::Psi<std::complex<double>> &hvchi = j2chi;
    psi::Psi<std::complex<double>> &vchi = j1chi;
    psi::Psi<std::complex<double>> &vhchi = tmpvchi;
    psi::Psi<std::complex<double>> &hchi = tmphchi;
    stoiter.stohchi.hchi(chi.get_pointer(), hchi.get_pointer(), perbands);
    velop.act(&chi, perbands, hchi.get_pointer(), vhchi.get_pointer());
    velop.act(&chi, perbands, chi.get_pointer(), vchi.get_pointer());
    stoiter.stohchi.hchi(vchi.get_pointer(), hvchi.get_pointer(), perbands*3);
    for(int ib = 0; ib < perbands*3; ++ib)
    {
        for(int ig = 0; ig < npw; ++ig)
        {
            j2chi(0, ib, ig) = 0.5*( hvchi(0, ib, ig) + vhchi(0, ib, ig)) - mu * vchi(0, ib, ig);
        }
    }
}

void ESolver_SDFT_PW::sKG(const int nche_KG,
                          const double fwhmin,
                          const double wcut,
                          const double dw_in,
                          const double dt_in,
                          const int nbatch,
                          const int npart_sto)
{
    ModuleBase::TITLE(this->classname, "sKG");
    ModuleBase::timer::tick(this->classname, "sKG");
    std::cout << "Calculating conductivity...." << std::endl;
    if (nbatch > 1)
    {
        ModuleBase::WARNING_QUIT("ESolver_SDFT_PW", "nbatch > 1!");
    }
    //------------------------------------------------------------------
    //                    Init
    //------------------------------------------------------------------
    // Parameters
    int nw = ceil(wcut / dw_in);
    double dw = dw_in / ModuleBase::Ry_to_eV; // converge unit in eV to Ry
    double sigma = fwhmin / TWOSQRT2LN2 / ModuleBase::Ry_to_eV;
    double dt = dt_in;                               // unit in a.u., 1 a.u. = 4.837771834548454e-17 s
    const double expfactor = 18.42;                  // exp(-18.42) = 1e-8
    int nt = ceil(sqrt(2 * expfactor) / sigma / dt); // set nt empirically
    std::cout << "nw: " << nw << " ; dw: " << dw * ModuleBase::Ry_to_eV << " eV" << std::endl;
    std::cout << "nt: " << nt << " ; dt: " << dt << " a.u.(ry^-1)" << std::endl;
    assert(nw >= 1);
    assert(nt >= 1);
    const int ndim = 3;
    const int nk = kv.nks;
    const int npwx = wf.npwx;
    const double tpiba = GlobalC::ucell.tpiba;
    psi::Psi<std::complex<double>>* stopsi;
    if (GlobalV::NBANDS > 0)
    {
        stopsi = stowf.chiortho;
        // clean memories //Note shchi is different from \sqrt(fH_here)|chi>, since veffs are different
        stowf.shchi->resize(1, 1, 1);
        stowf.chi0->resize(1, 1, 1); // clean memories
    }
    else
    {
        stopsi = stowf.chi0;
        stowf.shchi->resize(1, 1, 1); // clean memories
    }
    const double dEcut = (wcut + fwhmin) / ModuleBase::Ry_to_eV;

    // response funtion
    double* ct11 = new double[nt];
    double* ct12 = new double[nt];
    double* ct22 = new double[nt];
    ModuleBase::GlobalFunc::ZEROS(ct11, nt);
    ModuleBase::GlobalFunc::ZEROS(ct12, nt);
    ModuleBase::GlobalFunc::ZEROS(ct22, nt);

    // Init Chebyshev
    ModuleBase::Chebyshev<double> che(this->nche_sto);
    ModuleBase::Chebyshev<double> chemt(nche_KG);
    Stochastic_Iter& stoiter = ((hsolver::HSolverPW_SDFT*)phsol)->stoiter;
    Stochastic_hchi& stohchi = stoiter.stohchi;

    //------------------------------------------------------------------
    //                    Calculate
    //------------------------------------------------------------------

    // Prepare Chebyshev coefficients for exp(-i H/\hbar t)
    const double mu = this->pelec->eferm.ef;
    stoiter.stofunc.mu = mu;
    stoiter.stofunc.t = dt * nbatch;
    chemt.calcoef_pair(&stoiter.stofunc, &Sto_Func<double>::ncos, &Sto_Func<double>::n_sin);
    std::complex<double>*batchcoef = nullptr, *batchmcoef = nullptr;
    // if (nbatch > 1)
    // {
    //     batchcoef = new std::complex<double>[nche_KG * nbatch];
    //     std::complex<double>* tmpcoef = batchcoef + (nbatch - 1) * nche_KG;
    //     batchmcoef = new std::complex<double>[nche_KG * nbatch];
    //     std::complex<double>* tmpmcoef = batchmcoef + (nbatch - 1) * nche_KG;
    //     for (int i = 0; i < nche_KG; ++i)
    //     {
    //         tmpcoef[i] = chet.coef_complex[i];
    //         tmpmcoef[i] = chemt.coef_complex[i];
    //     }
    //     for (int ib = 0; ib < nbatch - 1; ++ib)
    //     {
    //         tmpcoef = batchcoef + ib * nche_KG;
    //         tmpmcoef = batchmcoef + ib * nche_KG;
    //         stoiter.stofunc.t = 0.5 * dt * (ib + 1);
    //         chet.calcoef_pair(&stoiter.stofunc, &Sto_Func<double>::ncos, &Sto_Func<double>::nsin);
    //         chemt.calcoef_pair(&stoiter.stofunc, &Sto_Func<double>::ncos, &Sto_Func<double>::n_sin);
    //         for (int i = 0; i < nche_KG; ++i)
    //         {
    //             tmpcoef[i] = chet.coef_complex[i];
    //             tmpmcoef[i] = chemt.coef_complex[i];
    //         }
    //     }
    //     stoiter.stofunc.t = 0.5 * dt * nbatch;
    // }

    // ik loop
    ModuleBase::timer::tick(this->classname, "kloop");
    hamilt::Velocity velop(pw_wfc, kv.isk.data(), &GlobalC::ppcell, &GlobalC::ucell, INPUT.cond_nonlocal);
    for (int ik = 0; ik < nk; ++ik)
    {
        velop.init(ik);
        stopsi->fix_k(ik);
        psi->fix_k(ik);
        if (nk > 1)
        {
            this->p_hamilt->updateHk(ik);
        }
        stoiter.stohchi.current_ik = ik;
        const int npw = kv.ngk[ik];

        double Emin = stoiter.stofunc.Emin;
        if (GlobalV::NBANDS > 0)
        {
            Emin = this->pelec->ekb(ik, 0);
        }
        double dE = stoiter.stofunc.Emax - Emin + wcut / ModuleBase::Ry_to_eV;
        std::cout << "Emin: " << Emin * ModuleBase::Ry_to_eV
                  << " eV; Emax: " << stoiter.stofunc.Emax * ModuleBase::Ry_to_eV
                  << " eV; Recommended dt: " << 2 * M_PI / dE << " a.u." << std::endl;

        // Parallel for bands
        int allbands_ks = GlobalV::NBANDS;
        parallel_distribution paraks(allbands_ks, GlobalV::NSTOGROUP, GlobalV::MY_STOGROUP);
        int perbands_ks = paraks.num_per;
        int ib0_ks = paraks.start;
        int perbands_sto = this->stowf.nchip[ik];
        int perbands = perbands_sto + perbands_ks;

        //-------------------     allocate  -------------------------
        size_t memory_cost = perbands * npwx * sizeof(std::complex<double>);
        psi::Psi<std::complex<double>> right(1, perbands, npwx, kv.ngk.data());
        ModuleBase::Memory::record("SDFT::right", memory_cost);
        
        //prepare |right>
        for (int ib = 0; ib < perbands_ks; ++ib)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                right(0,ib,ig) = psi[0](ib0_ks + ib, ig);
            }
        }
        for (int ib = 0; ib < perbands_sto; ++ib)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                right(0, ib + perbands_ks, ig) = stopsi[0](ib, ig);
            }
        }

        //sqrt(f)|right>
        psi::Psi<std::complex<double>>& sfright = right;
        che.calcoef_real(&stoiter.stofunc, &Sto_Func<double>::nroot_fd);
        che.calfinalvec_real(&stohchi,
                             &Stochastic_hchi::hchi_norm,
                             right.get_pointer(),
                             sfright.get_pointer(),
                             npw,
                             npwx,
                             perbands);
        
        //prepare sqrt(f)|ileft>
        psi::Psi<std::complex<double>> sfleft(1, perbands, npwx, kv.ngk.data());
        ModuleBase::Memory::record("SDFT::left", memory_cost);
        for(int ib = 0; ib < perbands; ++ib)
        {
            for(int ig = 0; ig < npw; ++ig)
            {
                sfleft(0, ib, ig) = ModuleBase::IMAG_UNIT * sfright(0, ib, ig);
            }
        }
        
        psi::Psi<std::complex<double>> j1right(1, perbands*3, npwx, kv.ngk.data());
        psi::Psi<std::complex<double>> j2right(1, perbands*3, npwx, kv.ngk.data());
        psi::Psi<std::complex<double>> j1left(1, perbands*3, npwx, kv.ngk.data());
        psi::Psi<std::complex<double>> j2left(1, perbands*3, npwx, kv.ngk.data());
        psi::Psi<std::complex<double>> tmpvchi(1, perbands*3, npwx, kv.ngk.data());
        psi::Psi<std::complex<double>> tmphchi(1, perbands, npwx, kv.ngk.data());
        ModuleBase::Memory::record("SDFT::vchi", memory_cost*16);
        
        // J|sfileft>
        this->calj12(ik, perbands, sfleft, j1left, j2left, tmpvchi, tmphchi, velop);

        // (1-f)J|sfileft>
        che.calcoef_real(&stoiter.stofunc, &Sto_Func<double>::nroot_mfd);
        che.calfinalvec_real(&stohchi,
                             &Stochastic_hchi::hchi_norm,
                             j1left.get_pointer(),
                             j1left.get_pointer(),
                             npw,
                             npwx,
                             perbands*3);
        che.calfinalvec_real(&stohchi,
                             &Stochastic_hchi::hchi_norm,
                             j1left.get_pointer(),
                             j1left.get_pointer(),
                             npw,
                             npwx,
                             perbands*3);
        che.calfinalvec_real(&stohchi,
                             &Stochastic_hchi::hchi_norm,
                             j2left.get_pointer(),
                             j2left.get_pointer(),
                             npw,
                             npwx,
                             perbands*3);
        che.calfinalvec_real(&stohchi,
                             &Stochastic_hchi::hchi_norm,
                             j2left.get_pointer(),
                             j2left.get_pointer(),
                             npw,
                             npwx,
                             perbands*3);
        
        //------------------------  allocate ------------------------
        
        // if (nbatch > 1)
        // {
        //     poly_exptsfchi.resize(nche_KG, perbands_sto, npwx);
        //     ModuleBase::Memory::record("SDFT::poly_exptsfchi",
        //                                sizeof(std::complex<double>) * nche_KG * perbands_sto * npwx);

        //     poly_exptsmfchi.resize(nche_KG, perbands_sto, npwx);
        //     ModuleBase::Memory::record("SDFT::poly_exptsmfchi",
        //                                sizeof(std::complex<double>) * nche_KG * perbands_sto * npwx);

        //     poly_expmtsfchi.resize(nche_KG, perbands_sto, npwx);
        //     ModuleBase::Memory::record("SDFT::poly_expmtsfchi",
        //                                sizeof(std::complex<double>) * nche_KG * perbands_sto * npwx);

        //     poly_expmtsmfchi.resize(nche_KG, perbands_sto, npwx);
        //     ModuleBase::Memory::record("SDFT::poly_expmtsmfchi",
        //                                sizeof(std::complex<double>) * nche_KG * perbands_sto * npwx);
        // }

        //------------------------  t loop  --------------------------
        std::cout << "ik=" << ik << ": ";
        auto start = std::chrono::high_resolution_clock::now();
        const int print_step = ceil(20.0 / nbatch) * nbatch;
        for (int it = 1; it < nt; ++it)
        {
            // evaluate time cost
            if (it - 1 == print_step)
            {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration = end - start;
                double timeTaken = duration.count();
                std::cout << "(Time left " << timeTaken * (double(nt - 1) / print_step * (nk - ik) - 1) << " s) "
                          << std::endl;
                std::cout << "nt: "<<std::endl;
            }
            if ((it - 1) % print_step == 0 && it > 1)
            {
                std::cout <<std::setw(8)<< it - 1;
                if( (it - 1)% (print_step*10) == 0)
                {
                    std::cout << std::endl;
                }
            }

            ModuleBase::timer::tick(this->classname, "evolution");
            // Sto
            // if (nbatch == 1)
            // {
            chemt.calfinalvec_complex(&stohchi,
                                      &Stochastic_hchi::hchi_norm,
                                      sfright.get_pointer(),
                                      sfright.get_pointer(),
                                      npw,
                                      npwx,
                                      perbands);
            chemt.calfinalvec_complex(&stohchi,
                                     &Stochastic_hchi::hchi_norm,
                                     j1left.get_pointer(),
                                     j1left.get_pointer(),
                                     npw,
                                     npwx,
                                     perbands*3);
            chemt.calfinalvec_complex(&stohchi,
                                     &Stochastic_hchi::hchi_norm,
                                     j2left.get_pointer(),
                                     j2left.get_pointer(),
                                     npw,
                                     npwx,
                                     perbands*3);
            this->calj12(ik, perbands, sfright, j1right, j2right, tmpvchi, tmphchi, velop);
            
            ////Im<left|right>=Re<left|-i|right>=Re<ileft|J|right>
            for(int ib = 0 ; ib < perbands*3 ; ++ib)
            {
                
                ct11[it] += ModuleBase::GlobalFunc::ddot_real(npw,&j1left(0,ib,0), &j1right(0,ib,0),false) * kv.wk[ik] / 2,0;
                double tmp12
                    = ModuleBase::GlobalFunc::ddot_real(npw,&j1left(0,ib,0),&j2right(0,ib,0),false);
                double tmp21
                    = ModuleBase::GlobalFunc::ddot_real(npw,&j2left(0,ib,0),&j1right(0,ib,0),false);
                ct12[it] -= 0.5*(tmp12 + tmp21) * kv.wk[ik] / 2.0;
                // std::cout<<tmp12<<" "<<tmp21<<std::endl;
                ct22[it] += ModuleBase::GlobalFunc::ddot_real(npw,&j2left(0,ib,0),&j2right(0,ib,0),false) * kv.wk[ik] / 2.0;
            }
            // }
            
            
            
            // else
            // {
            //     std::complex<double>* tmppolyexpmtsfchi = poly_expmtsfchi.get_pointer();
            //     std::complex<double>* tmppolyexpmtsmfchi = poly_expmtsmfchi.get_pointer();
            //     std::complex<double>* tmppolyexptsfchi = poly_exptsfchi.get_pointer();
            //     std::complex<double>* tmppolyexptsmfchi = poly_exptsmfchi.get_pointer();
            //     std::complex<double>* stoexpmtsfchi = expmtsfchi.get_pointer();
            //     std::complex<double>* stoexpmtsmfchi = expmtsmfchi.get_pointer();
            //     std::complex<double>* stoexptsfchi = exptsfchi.get_pointer();
            //     std::complex<double>* stoexptsmfchi = exptsmfchi.get_pointer();
            //     if ((it - 1) % nbatch == 0)
            //     {
            //         chet.calpolyvec_complex(&stohchi,
            //                                 &Stochastic_hchi::hchi_norm,
            //                                 stoexptsfchi,
            //                                 tmppolyexptsfchi,
            //                                 npw,
            //                                 npwx,
            //                                 perbands_sto);
            //     }

            //     std::complex<double>* tmpcoef = batchcoef + (it - 1) % nbatch * nche_KG;
            //     std::complex<double>* tmpmcoef = batchmcoef + (it - 1) % nbatch * nche_KG;
            //     const char transa = 'N';
            //     const int LDA = perbands_sto * npwx;
            //     const int M = perbands_sto * npwx;
            //     const int N = nche_KG;
            //     const int inc = 1;
            //     zgemv_(&transa,
            //            &M,
            //            &N,
            //            &ModuleBase::ONE,
            //            tmppolyexptsfchi,
            //            &LDA,
            //            tmpcoef,
            //            &inc,
            //            &ModuleBase::ZERO,
            //            stoexptsfchi,
            //            &inc);
            // }
            ModuleBase::timer::tick(this->classname, "evolution");
        }
        std::cout << std::endl;
    } // ik loop
    ModuleBase::timer::tick(this->classname, "kloop");
    delete[] batchcoef;
    delete[] batchmcoef;

#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, ct11, nt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ct12, nt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ct22, nt, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    //------------------------------------------------------------------
    //                    Output
    //------------------------------------------------------------------
    if (GlobalV::MY_RANK == 0)
    {
        calcondw(nt, dt, fwhmin, wcut, dw_in, ct11, ct12, ct22);
    }
    delete[] ct11;
    delete[] ct12;
    delete[] ct22;
    ModuleBase::timer::tick(this->classname, "sKG");
}

void ESolver_SDFT_PW::caldos(const int nche_dos,
                             const double sigmain,
                             const double emin,
                             const double emax,
                             const double de,
                             const int npart)
{
    ModuleBase::TITLE(this->classname, "caldos");
    ModuleBase::timer::tick(this->classname, "caldos");
    std::cout << "=========================" << std::endl;
    std::cout << "###Calculating Dos....###" << std::endl;
    std::cout << "=========================" << std::endl;
    ModuleBase::Chebyshev<double> che(nche_dos);
    const int nk = kv.nks;
    Stochastic_Iter& stoiter = ((hsolver::HSolverPW_SDFT*)phsol)->stoiter;
    Stochastic_hchi& stohchi = stoiter.stohchi;
    const int npwx = wf.npwx;

    double* spolyv = nullptr;
    std::complex<double>* allorderchi = nullptr;
    if (stoiter.method == 1)
    {
        spolyv = new double[nche_dos];
        ModuleBase::GlobalFunc::ZEROS(spolyv, nche_dos);
    }
    else
    {
        spolyv = new double[nche_dos * nche_dos];
        ModuleBase::GlobalFunc::ZEROS(spolyv, nche_dos * nche_dos);
        int nchip_new = ceil((double)this->stowf.nchip_max / npart);
        allorderchi = new std::complex<double>[nchip_new * npwx * nche_dos];
    }
    ModuleBase::timer::tick(this->classname, "Tracepoly");
    std::cout << "1. TracepolyA:" << std::endl;
    for (int ik = 0; ik < nk; ik++)
    {
        std::cout << "ik: " << ik + 1 << std::endl;
        if (nk > 1)
        {
            this->p_hamilt->updateHk(ik);
        }
        stohchi.current_ik = ik;
        const int npw = kv.ngk[ik];
        const int nchipk = this->stowf.nchip[ik];

        std::complex<double>* pchi;
        if (GlobalV::NBANDS > 0)
        {
            stowf.chiortho->fix_k(ik);
            pchi = stowf.chiortho->get_pointer();
        }
        else
        {
            stowf.chi0->fix_k(ik);
            pchi = stowf.chi0->get_pointer();
        }
        if (stoiter.method == 1)
        {
            che.tracepolyA(&stohchi, &Stochastic_hchi::hchi_norm, pchi, npw, npwx, nchipk);
            for (int i = 0; i < nche_dos; ++i)
            {
                spolyv[i] += che.polytrace[i] * kv.wk[ik] / 2;
            }
        }
        else
        {
            int N = nche_dos;
            double kweight = kv.wk[ik] / 2;
            char trans = 'T';
            char normal = 'N';
            double one = 1;
            for (int ipart = 0; ipart < npart; ++ipart)
            {
                int nchipk_new = nchipk / npart;
                int start_nchipk = ipart * nchipk_new + nchipk % npart;
                if (ipart < nchipk % npart)
                {
                    nchipk_new++;
                    start_nchipk = ipart * nchipk_new;
                }
                ModuleBase::GlobalFunc::ZEROS(allorderchi, nchipk_new * npwx * nche_dos);
                std::complex<double>* tmpchi = pchi + start_nchipk * npwx;
                che.calpolyvec_complex(&stohchi,
                                       &Stochastic_hchi::hchi_norm,
                                       tmpchi,
                                       allorderchi,
                                       npw,
                                       npwx,
                                       nchipk_new);
                double* vec_all = (double*)allorderchi;
                int LDA = npwx * nchipk_new * 2;
                int M = npwx * nchipk_new * 2;
                dgemm_(&trans, &normal, &N, &N, &M, &kweight, vec_all, &LDA, vec_all, &LDA, &one, spolyv, &N);
            }
        }
    }
    if (stoiter.method == 2)
        delete[] allorderchi;

    std::ofstream ofsdos;
    int ndos = int((emax - emin) / de) + 1;
    stoiter.stofunc.sigma = sigmain / ModuleBase::Ry_to_eV;
    ModuleBase::timer::tick(this->classname, "Tracepoly");

    std::cout << "2. Dos:" << std::endl;
    ModuleBase::timer::tick(this->classname, "DOS Loop");
    int n10 = ndos / 10;
    int percent = 10;
    double* sto_dos = new double[ndos];
    double* ks_dos = new double[ndos];
    double* error = new double[ndos];
    for (int ie = 0; ie < ndos; ++ie)
    {
        double tmpks = 0;
        double tmpsto = 0;
        stoiter.stofunc.targ_e = (emin + ie * de) / ModuleBase::Ry_to_eV;
        if (stoiter.method == 1)
        {
            che.calcoef_real(&stoiter.stofunc, &Sto_Func<double>::ngauss);
            tmpsto = BlasConnector::dot(nche_dos, che.coef_real, 1, spolyv, 1);
        }
        else
        {
            che.calcoef_real(&stoiter.stofunc, &Sto_Func<double>::nroot_gauss);
            tmpsto = stoiter.vTMv(che.coef_real, spolyv, nche_dos);
        }
        if (GlobalV::NBANDS > 0)
        {
            for (int ik = 0; ik < nk; ++ik)
            {
                double* en = &(this->pelec->ekb(ik, 0));
                for (int ib = 0; ib < GlobalV::NBANDS; ++ib)
                {
                    tmpks += stoiter.stofunc.gauss(en[ib]) * kv.wk[ik] / 2;
                }
            }
        }
        tmpks /= GlobalV::NPROC_IN_POOL;

        double tmperror = 0;
        if (stoiter.method == 1)
        {
            tmperror = che.coef_real[nche_dos - 1] * spolyv[nche_dos - 1];
        }
        else
        {
            const int norder = nche_dos;
            double last_coef = che.coef_real[norder - 1];
            double last_spolyv = spolyv[norder * norder - 1];
            tmperror = last_coef
                       * (BlasConnector::dot(norder, che.coef_real, 1, spolyv + norder * (norder - 1), 1)
                          + BlasConnector::dot(norder, che.coef_real, 1, spolyv + norder - 1, norder)
                          - last_coef * last_spolyv);
        }

        if (ie % n10 == n10 - 1)
        {
            std::cout << percent << "%"
                      << " ";
            percent += 10;
        }
        sto_dos[ie] = tmpsto;
        ks_dos[ie] = tmpks;
        error[ie] = tmperror;
    }
#ifdef __MPI
    MPI_Allreduce(MPI_IN_PLACE, ks_dos, ndos, MPI_DOUBLE, MPI_SUM, STO_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sto_dos, ndos, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, error, ndos, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (GlobalV::MY_RANK == 0)
    {
        std::string dosfile = GlobalV::global_out_dir + "DOS1_smearing.dat";
        ofsdos.open(dosfile.c_str());
        double maxerror = 0;
        double sum = 0;
        ofsdos << std::setw(8) << "## E(eV) " << std::setw(20) << "dos(eV^-1)" << std::setw(20) << "sum"
               << std::setw(20) << "Error(eV^-1)" << std::endl;
        for (int ie = 0; ie < ndos; ++ie)
        {
            double tmperror = 2.0 * std::abs(error[ie]);
            if (maxerror < tmperror)
                maxerror = tmperror;
            double dos = 2.0 * (ks_dos[ie] + sto_dos[ie]) / ModuleBase::Ry_to_eV;
            sum += dos;
            ofsdos << std::setw(8) << emin + ie * de << std::setw(20) << dos << std::setw(20) << sum * de
                   << std::setw(20) << tmperror << std::endl;
        }
        std::cout << std::endl;
        std::cout << "Finish DOS" << std::endl;
        std::cout << std::scientific << "DOS max absolute Chebyshev Error: " << maxerror << std::endl;
        ofsdos.close();
    }
    delete[] sto_dos;
    delete[] ks_dos;
    delete[] error;
    delete[] spolyv;
    ModuleBase::timer::tick(this->classname, "DOS Loop");
    ModuleBase::timer::tick(this->classname, "caldos");
    return;
}

} // namespace ModuleESolver

namespace GlobalTemp
{

const ModuleBase::matrix* veff;

}