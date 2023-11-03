#include "sto_hchi.h"

#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_base/tool_title.h"
#include "module_esolver/esolver_sdft_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

Stochastic_hchi::Stochastic_hchi()
{
    Emin = INPUT.emin_sto;
    Emax = INPUT.emax_sto;
}

Stochastic_hchi::~Stochastic_hchi()
{
    delete[] f_king;
    delete[] f_veff;
    delete[] f_vkb;
    delete[] f_deeq;
}

void Stochastic_hchi::init(ModulePW::PW_Basis_K* wfc_basis, K_Vectors* pkv_in)
{
    wfcpw = wfc_basis;
    pkv = pkv_in;
}

void Stochastic_hchi::updateik_float(const int& ik)
{
    // constants
    const int npwx = this->wfcpw->npwk_max;
    const int npw = this->wfcpw->npwk[ik];
    const double tpiba2 = GlobalC::ucell.tpiba2;
    const int nrxx = this->wfcpw->nrxx;
    const int current_spin = pkv->isk[ik];

    // init ik
    this->current_ik = ik;

    // init f_king
    delete[] f_king;
    f_king = new float[npw];
    for (int ig = 0; ig < npw; ++ig)
    {
        f_king[ig] = static_cast<float>(this->wfcpw->getgk2(ik, ig) * tpiba2);
    }

    // init f_veff
    delete[] f_veff;
    f_veff = new float[nrxx];
    for (int ir = 0; ir < nrxx; ++ir)
    {
        f_veff[ir] = static_cast<float>((*GlobalTemp::veff)(current_spin, ir));
    }

    // init f_vkb && f_deeq
    if (GlobalV::VNL_IN_H && GlobalC::ppcell.nkb > 0)
    {
        const int nkb = GlobalC::ppcell.nkb;
        delete[] f_vkb;
        f_vkb = new std::complex<float>[nkb * npw];

        int dim_deeq = 0;
        for (int it = 0; it < GlobalC::ucell.ntype; it++)
        {
            const int Nprojs = GlobalC::ucell.atoms[it].ncpp.nh;
            dim_deeq += Nprojs * Nprojs * GlobalC::ucell.atoms[it].na;
        }
        delete[] f_deeq;
        f_deeq = new float[dim_deeq];

        std::complex<double> *vkb = GlobalC::ppcell.vkb.c;
        for (int ikb = 0; ikb < nkb; ++ikb)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                f_vkb[ikb * npw + ig] = static_cast<std::complex<float>>(vkb[ikb * npwx + ig]);
            }
        }
        int ipp = 0;
        int iat = 0;
        for (int it = 0; it < GlobalC::ucell.ntype; it++)
        {
            const int Nprojs = GlobalC::ucell.atoms[it].ncpp.nh;
            const int Nprojs2 = Nprojs * Nprojs;
            for (int ip = 0; ip < Nprojs; ip++)
            {
                for (int ip2 = 0; ip2 < Nprojs; ip2++)
                {
                    f_deeq[ipp + ip * Nprojs + ip2]
                        = static_cast<float>(GlobalC::ppcell.deeq(current_spin, iat, ip, ip2));
                }
            }
			for (int ia = 1; ia < GlobalC::ucell.atoms[it].na; ia++)
			{
				for (int ip = 0; ip < Nprojs; ip++)
            	{
            	    for (int ip2 = 0; ip2 < Nprojs; ip2++)
            	    {
            	            assert(GlobalC::ppcell.deeq(current_spin, iat + ia, ip, ip2) == GlobalC::ppcell.deeq(current_spin, iat, ip, ip2));
            	    }
            	}
			}
            iat += GlobalC::ucell.atoms[it].na;
            ipp += Nprojs2;
        }
    }

    return;
}

void Stochastic_hchi::hchi(complex<double>* chig, complex<double>* hchig, const int m)
{

    //---------------------------------------------------

    const int ik = this->current_ik;
    const int current_spin = pkv->isk[ik];
    const int npwx = this->wfcpw->npwk_max;
    const int npw = this->wfcpw->npwk[ik];
    const int npm = GlobalV::NPOL * m;
    const int inc = 1;
    const double tpiba2 = GlobalC::ucell.tpiba2;
    const int nrxx = this->wfcpw->nrxx;
    //------------------------------------
    //(1) the kinetical energy.
    //------------------------------------
    complex<double>* chibg = chig;
    complex<double>* hchibg = hchig;
    if (GlobalV::T_IN_H)
    {
        for (int ib = 0; ib < m; ++ib)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                hchibg[ig] = this->wfcpw->getgk2(ik, ig) * tpiba2 * chibg[ig];
            }
            chibg += npwx;
            hchibg += npwx;
        }
    }

    //------------------------------------
    //(2) the local potential.
    //------------------------------------
    ModuleBase::timer::tick("Stochastic_hchi", "vloc");
    std::complex<double>* porter = new std::complex<double>[nrxx];
    if (GlobalV::VL_IN_H)
    {
        chibg = chig;
        hchibg = hchig;
        const double* pveff = &((*GlobalTemp::veff)(current_spin, 0));
        for (int ib = 0; ib < m; ++ib)
        {
            this->wfcpw->recip2real(chibg, porter, ik);
            for (int ir = 0; ir < nrxx; ir++)
            {
                porter[ir] *= pveff[ir];
            }
            this->wfcpw->real2recip(porter, hchibg, ik, true);

            chibg += npwx;
            hchibg += npwx;
        }
    }
    delete[] porter;
    ModuleBase::timer::tick("Stochastic_hchi", "vloc");

    //------------------------------------
    // (3) the nonlocal pseudopotential.
    //------------------------------------
    ModuleBase::timer::tick("Stochastic_hchi", "vnl");
    if (GlobalV::VNL_IN_H)
    {
        if (GlobalC::ppcell.nkb > 0)
        {
            int nkb = GlobalC::ppcell.nkb;
            complex<double>* becp = new complex<double>[nkb * GlobalV::NPOL * m];
            char transc = 'C';
            char transn = 'N';
            char transt = 'T';
            if (m == 1 && GlobalV::NPOL == 1)
            {
                zgemv_(&transc,
                       &npw,
                       &nkb,
                       &ModuleBase::ONE,
                       GlobalC::ppcell.vkb.c,
                       &npwx,
                       chig,
                       &inc,
                       &ModuleBase::ZERO,
                       becp,
                       &inc);
            }
            else
            {
                zgemm_(&transc,
                       &transn,
                       &nkb,
                       &npm,
                       &npw,
                       &ModuleBase::ONE,
                       GlobalC::ppcell.vkb.c,
                       &npwx,
                       chig,
                       &npwx,
                       &ModuleBase::ZERO,
                       becp,
                       &nkb);
            }
            Parallel_Reduce::reduce_pool(becp, nkb * GlobalV::NPOL * m);

            complex<double>* Ps = new complex<double>[nkb * GlobalV::NPOL * m];
            ModuleBase::GlobalFunc::ZEROS(Ps, GlobalV::NPOL * m * nkb);

            int sum = 0;
            int iat = 0;
            for (int it = 0; it < GlobalC::ucell.ntype; it++)
            {
                const int Nprojs = GlobalC::ucell.atoms[it].ncpp.nh;
                for (int ia = 0; ia < GlobalC::ucell.atoms[it].na; ia++)
                {
                    // each atom has Nprojs, means this is with structure factor;
                    // each projector (each atom) must multiply coefficient
                    // with all the other projectors.
                    for (int ip = 0; ip < Nprojs; ip++)
                    {
                        for (int ip2 = 0; ip2 < Nprojs; ip2++)
                        {
                            for (int ib = 0; ib < m; ++ib)
                            {
                                Ps[(sum + ip2) * m + ib]
                                    += GlobalC::ppcell.deeq(current_spin, iat, ip, ip2) * becp[ib * nkb + sum + ip];
                            } // end ib
                        }     // end ih
                    }         // end jh
                    sum += Nprojs;
                    ++iat;
                } // end na
            }     // end nt

            if (GlobalV::NPOL == 1 && m == 1)
            {
                zgemv_(&transn,
                       &npw,
                       &nkb,
                       &ModuleBase::ONE,
                       GlobalC::ppcell.vkb.c,
                       &npwx,
                       Ps,
                       &inc,
                       &ModuleBase::ONE,
                       hchig,
                       &inc);
            }
            else
            {
                zgemm_(&transn,
                       &transt,
                       &npw,
                       &npm,
                       &nkb,
                       &ModuleBase::ONE,
                       GlobalC::ppcell.vkb.c,
                       &npwx,
                       Ps,
                       &npm,
                       &ModuleBase::ONE,
                       hchig,
                       &npwx);
            }

            delete[] becp;
            delete[] Ps;
        }
    }
    ModuleBase::timer::tick("Stochastic_hchi", "vnl");

    return;
}

void Stochastic_hchi::hchi(complex<float>* chig, complex<float>* hchig, const int m)
{
    ModuleBase::timer::tick("Stochastic_hchi", "hchi_float");

    //---------------------------------------------------

    const int ik = this->current_ik;
    const int current_spin = pkv->isk[ik];
    const int npwx = this->wfcpw->npwk_max;
    const int npw = this->wfcpw->npwk[ik];
    const int npm = GlobalV::NPOL * m;
    const int inc = 1;
    const int nrxx = this->wfcpw->nrxx;
    const std::complex<float> fone = 1.0f;
    const std::complex<float> fzero = 0.0f;
    //------------------------------------
    //(1) the kinetical energy.
    //------------------------------------
    complex<float>* chibg = chig;
    complex<float>* hchibg = hchig;

    for (int ib = 0; ib < m; ++ib)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            hchibg[ig] = this->f_king[ig] * chibg[ig];
        }
        chibg += npwx;
        hchibg += npwx;
    }

    //------------------------------------
    //(2) the local potential.
    //------------------------------------
    std::complex<float>* porter = new std::complex<float>[nrxx];
    chibg = chig;
    hchibg = hchig;
    for (int ib = 0; ib < m; ++ib)
    {
        this->wfcpw->recip2real(chibg, porter, ik);
        for (int ir = 0; ir < nrxx; ir++)
        {
            porter[ir] *= this->f_veff[ir];
        }
        this->wfcpw->real2recip(porter, hchibg, ik, true);
        chibg += npwx;
        hchibg += npwx;
    }
    delete[] porter;

    //------------------------------------
    // (3) the nonlocal pseudopotential.
    //------------------------------------
    if (GlobalC::ppcell.nkb > 0)
    {
        int nkb = GlobalC::ppcell.nkb;
        complex<float>* becp = new complex<float>[nkb * GlobalV::NPOL * m];
        char transc = 'C';
        char transn = 'N';
        char transt = 'T';
        if (m == 1 && GlobalV::NPOL == 1)
        {
            cgemv_(&transc, &npw, &nkb, &fone, f_vkb, &npw, chig, &inc, &fzero, becp, &inc);
        }
        else
        {
            cgemm_(&transc, &transn, &nkb, &npm, &npw, &fone, f_vkb, &npw, chig, &npwx, &fzero, becp, &nkb);
        }
        Parallel_Reduce::reduce_pool(becp, nkb * GlobalV::NPOL * m);
        complex<float>* Ps = new complex<float>[nkb * GlobalV::NPOL * m];
        ModuleBase::GlobalFunc::ZEROS(Ps, GlobalV::NPOL * m * nkb);

        int sum = 0;
		int ipp = 0;
        for (int it = 0; it < GlobalC::ucell.ntype; it++)
        {
            const int Nprojs = GlobalC::ucell.atoms[it].ncpp.nh;
			const int Nprojs2 = Nprojs * Nprojs;
			for (int ia = 0; ia < GlobalC::ucell.atoms[it].na; ia++)
            {
            	for (int ip = 0; ip < Nprojs; ip++)
            	{
            	    for (int ip2 = 0; ip2 < Nprojs; ip2++)
            	    {
            	        for (int ib = 0; ib < m; ++ib)
            	        {
            	            Ps[(sum + ip2) * m + ib]
            	                += this->f_deeq[ipp + ip * Nprojs + ip2] * becp[ib * nkb + sum + ip];
            	        }
            	    }
            	}
            	sum += Nprojs;
		    }
			ipp += Nprojs2;
        } 

        if (GlobalV::NPOL == 1 && m == 1)
        {
            cgemv_(&transn, &npw, &nkb, &fone, f_vkb, &npw, Ps, &inc, &fone, hchig, &inc);
        }
        else
        {
            cgemm_(&transn, &transt, &npw, &npm, &nkb, &fone, f_vkb, &npw, Ps, &npm, &fone, hchig, &npwx);
        }
        delete[] becp;
        delete[] Ps;
    }
    ModuleBase::timer::tick("Stochastic_hchi", "hchi_float");
    return;
}

void Stochastic_hchi::hchi_norm(complex<double>* chig, complex<double>* hchig, const int m)
{
    ModuleBase::timer::tick("Stochastic_hchi", "hchi_norm");

    this->hchi(chig, hchig, m);

    const int ik = this->current_ik;
    const int npwx = this->wfcpw->npwk_max;
    const int npw = this->wfcpw->npwk[ik];
    const double Ebar = (Emin + Emax) / 2;
    const double DeltaE = (Emax - Emin) / 2;
    for (int ib = 0; ib < m; ++ib)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            hchig[ib * npwx + ig] = (hchig[ib * npwx + ig] - Ebar * chig[ib * npwx + ig]) / DeltaE;
        }
    }
    ModuleBase::timer::tick("Stochastic_hchi", "hchi_norm");
}

void Stochastic_hchi::hchi_norm(complex<float>* chig, complex<float>* hchig, const int m)
{
    ModuleBase::timer::tick("Stochastic_hchi", "hchi_norm_float");

    this->hchi(chig, hchig, m);

    const int ik = this->current_ik;
    const int npwx = this->wfcpw->npwk_max;
    const int npw = this->wfcpw->npwk[ik];
    const float Ebar = (Emin + Emax) / 2;
    const float DeltaE = (Emax - Emin) / 2;
    for (int ib = 0; ib < m; ++ib)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            hchig[ib * npwx + ig] = (hchig[ib * npwx + ig] - Ebar * chig[ib * npwx + ig]) / DeltaE;
        }
    }
    ModuleBase::timer::tick("Stochastic_hchi", "hchi_norm_float");
}