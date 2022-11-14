#include "velocity_pw.h"
#include "module_base/timer.h"
#include "src_parallel/parallel_reduce.h"
namespace hamilt
{

Velocity::Velocity
(
    const ModulePW::PW_Basis_K* wfcpw_in,
    const int* isk_in,
    pseudopot_cell_vnl* ppcell_in,
    const UnitCell_pseudo* ucell_in,
    const bool nonlocal_in
)
{
    this->wfcpw = wfcpw_in;
    this->isk = isk_in;
    this->ppcell = ppcell_in;
    this->ucell = ucell_in;
    this->nonlocal = nonlocal_in;
    if( this->wfcpw == nullptr || this->isk == nullptr || this->ppcell == nullptr || this->ucell == nullptr)
    {
        ModuleBase::WARNING_QUIT("Velocity", "Constuctor of Operator::Velocity is failed, please check your code!");
    }
    this->tpiba = ucell_in -> tpiba;
    if(this->nonlocal)      this->ppcell->initgradq_vnl(*this->ucell);
}

void Velocity::init(const int ik_in)
{
    this->ik = ik_in;
    // Calculate nonlocal pseudopotential vkb
	if(this->ppcell->nkb > 0 && this->nonlocal) 
	{
        this->ppcell->getgradq_vnl(ik_in);
	}

}

void Velocity::act
(
    const psi::Psi<std::complex<double>> *psi_in, 
    const int n_npwx, //nbands * NPOL
    const std::complex<double>* psi0, 
    std::complex<double>* vpsi,
    const bool add
) const
{
    ModuleBase::timer::tick("Operator", "Velocity");
    const int npw = psi_in->get_ngk(this->ik);
    const int max_npw = psi_in->get_nbasis() / psi_in->npol;
    const int npol = psi_in->npol;
    const std::complex<double>* tmpsi_in = psi0;
    std::complex<double>* tmhpsi = vpsi;
    // -------------
    //       p
    // -------------
    for (int ib = 0; ib < n_npwx; ++ib)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            ModuleBase::Vector3<double> tmpg = wfcpw->getgpluskcar(this->ik, ig);
            if(add)
            {
                tmhpsi[ig]                       += tmpsi_in[ig] * tmpg.x * tpiba;
                tmhpsi[ig + n_npwx * max_npw]    += tmpsi_in[ig] * tmpg.y * tpiba;
                tmhpsi[ig + 2 * n_npwx * max_npw]+= tmpsi_in[ig] * tmpg.z * tpiba;
            }
            else
            {
                tmhpsi[ig]                        = tmpsi_in[ig] * tmpg.x * tpiba;
                tmhpsi[ig + n_npwx * max_npw]     = tmpsi_in[ig] * tmpg.y * tpiba;
                tmhpsi[ig + 2 * n_npwx * max_npw] = tmpsi_in[ig] * tmpg.z * tpiba;
            }
        }
        tmhpsi += max_npw;
        tmpsi_in += max_npw;
    }

    // ---------------------------------------------
    // i[V_NL, r] = (\nabla_q+\nabla_q')V_{NL}(q,q') 
    // |\beta><\beta|\psi>
    // ---------------------------------------------
    if (this->ppcell->nkb <= 0 || !this->nonlocal) 
    {
        ModuleBase::timer::tick("Operator", "Velocity");
        return;
    }

    //1. <\beta|\psi>
    const int nkb = this->ppcell->nkb;
    const int nkb3 = 3 * nkb;
    ModuleBase::ComplexMatrix becp1(n_npwx, nkb, false);
    // ModuleBase::ComplexMatrix becp2(n_npwx, nkb3, false);
    ModuleBase::ComplexMatrix becp1_minus1(n_npwx, nkb, false);
    ModuleBase::ComplexMatrix becp1_plus1(n_npwx, nkb, false);
     ModuleBase::ComplexMatrix becp1_minus2(n_npwx, nkb, false);
    ModuleBase::ComplexMatrix becp1_plus2(n_npwx, nkb, false);
     ModuleBase::ComplexMatrix becp1_minus3(n_npwx, nkb, false);
    ModuleBase::ComplexMatrix becp1_plus3(n_npwx, nkb, false);
    char transC = 'C';
    char transN = 'N';
    char transT = 'T';
    const int npm = n_npwx;
    //-----------------------------------------------------
    const double thr = 1e-4;
    // ModuleBase::GlobalFunc::ZEROS(vpsi, 3 * n_npwx * max_npw);
    //-----------------------------------------------------

    if (n_npwx == 1)
    {
        ModuleBase::WARNING_QUIT("velocity_pw.cpp","Wrong");
    }
    else
    {
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1.c, &nkb);
        // zgemm_(&transC, &transN, &nkb3, &n_npwx, &npw,
        //        &ModuleBase::ONE, this->ppcell->gradvkb.ptr, &max_npw, psi0, &max_npw,
        //        &ModuleBase::ZERO, becp2.c, &nkb3);
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb_minus1.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1_minus1.c, &nkb);
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb_plus1.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1_plus1.c, &nkb);
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb_minus2.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1_minus2.c, &nkb);
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb_plus2.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1_plus2.c, &nkb);
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb_minus3.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1_minus3.c, &nkb);
        zgemm_(&transC, &transN, &nkb, &n_npwx, &npw,
               &ModuleBase::ONE, this->ppcell->vkb_plus3.c, &max_npw, psi0, &max_npw,
               &ModuleBase::ZERO, becp1_plus3.c, &nkb);
    }
    Parallel_Reduce::reduce_complex_double_pool(becp1.c, nkb * n_npwx);
    // Parallel_Reduce::reduce_complex_double_pool(becp2.c, nkb3 * n_npwx);
    Parallel_Reduce::reduce_complex_double_pool(becp1_minus1.c, nkb * n_npwx);
    Parallel_Reduce::reduce_complex_double_pool(becp1_plus1.c, nkb * n_npwx);
    Parallel_Reduce::reduce_complex_double_pool(becp1_minus2.c, nkb * n_npwx);
    Parallel_Reduce::reduce_complex_double_pool(becp1_plus2.c, nkb * n_npwx);
    Parallel_Reduce::reduce_complex_double_pool(becp1_minus3.c, nkb * n_npwx);
    Parallel_Reduce::reduce_complex_double_pool(becp1_plus3.c, nkb * n_npwx);

    //2. <\beta \psi><psi|
    ModuleBase::ComplexMatrix ps1(nkb, n_npwx, true);
    ModuleBase::ComplexMatrix ps1_minus1(nkb, n_npwx, true);
    ModuleBase::ComplexMatrix ps1_plus1(nkb, n_npwx, true);
    ModuleBase::ComplexMatrix ps1_minus2(nkb, n_npwx, true);
    ModuleBase::ComplexMatrix ps1_plus2(nkb, n_npwx, true);
    ModuleBase::ComplexMatrix ps1_minus3(nkb, n_npwx, true);
    ModuleBase::ComplexMatrix ps1_plus3(nkb, n_npwx, true);
    // ModuleBase::ComplexMatrix ps2(nkb3, n_npwx, true);

    int sum = 0;
    int iat = 0;
    if (npol == 1)
    {
        const int current_spin = this->isk[ik];
        for (int it = 0; it < this->ucell->ntype; it++)
        {
            const int nproj = this->ucell->atoms[it].nh;
            for (int ia = 0; ia < this->ucell->atoms[it].na; ia++)
            {
                for (int ip = 0; ip < nproj; ip++)
                {
                    for (int ip2 = 0; ip2 < nproj; ip2++)
                    {
                        for (int ib = 0; ib < n_npwx; ++ib)
                        {
                            double dij = this->ppcell->deeq(current_spin, iat, ip, ip2);
                            int sumip2 = sum + ip2;
                            int sumip = sum + ip;
                            ps1(sumip2, ib)  += dij * becp1(ib, sumip);
                            ps1_minus1(sumip2, ib)  += dij * becp1_minus1(ib, sumip);
                            ps1_plus1(sumip2, ib)  += dij * becp1_plus1(ib, sumip);
                            ps1_minus2(sumip2, ib)  += dij * becp1_minus2(ib, sumip);
                            ps1_plus2(sumip2, ib)  += dij * becp1_plus2(ib, sumip);
                            ps1_minus3(sumip2, ib)  += dij * becp1_minus3(ib, sumip);
                            ps1_plus3(sumip2, ib)  += dij * becp1_plus3(ib, sumip);
                            // ps2(sumip2, ib)  += dij * becp2(ib, sumip);
                            // ps2(sumip2 + nkb, ib)  += dij * becp2(ib, sumip + nkb);
                            // ps2(sumip2 + 2*nkb, ib)  += dij * becp2(ib , sumip + 2*nkb);
                        }
                    }
                }
                sum += nproj;
                ++iat;
            }
        }
    }
    else
    {
        ModuleBase::WARNING_QUIT("velocity_pw.cpp","Wrong");
    }

    // std::complex<double> *vpsi2 = new std::complex<double> [max_npw * n_npwx * 3];
    // ModuleBase::GlobalFunc::ZEROS(vpsi2, n_npwx * max_npw * 3);
    if (n_npwx == 1)
    {
        ModuleBase::WARNING_QUIT("velocity_pw.cpp","Wrong");
    }
    else
    {
        // for(int id = 0 ; id < 3 ; ++id)
        // {
        //     int vkbshift = id * max_npw * nkb;
        //     int ps2shift = id * n_npwx * nkb;
        //     int npwshift = id * max_npw * n_npwx;
        //     zgemm_(&transN, &transT, &npw, &npm, &nkb,
        //        &ModuleBase::ONE, this->ppcell->gradvkb.ptr + vkbshift, &max_npw, ps1.c, &n_npwx,
        //        &ModuleBase::ONE, vpsi + npwshift, &max_npw);
        //     zgemm_(&transN, &transT, &npw, &npm, &nkb,
        //        &ModuleBase::ONE, this->ppcell->vkb.c, &max_npw, ps2.c + ps2shift, &n_npwx,
        //        &ModuleBase::ONE, vpsi + npwshift, &max_npw);
        // }
        const std::complex<double> factor = 1.0/ (2.0 * thr * wfcpw->tpiba);
        const std::complex<double> negfactor = -factor;
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
           &negfactor, this->ppcell->vkb.c, &max_npw, ps1_minus1.c, &n_npwx,
           &ModuleBase::ONE, vpsi, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &negfactor, this->ppcell->vkb_minus1.c, &max_npw, ps1.c, &n_npwx,
               &ModuleBase::ONE, vpsi, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &factor, this->ppcell->vkb.c, &max_npw, ps1_plus1.c, &n_npwx,
               &ModuleBase::ONE, vpsi, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &factor, this->ppcell->vkb_plus1.c, &max_npw, ps1.c, &n_npwx,
               &ModuleBase::ONE, vpsi, &max_npw);

        zgemm_(&transN, &transT, &npw, &npm, &nkb,
           &negfactor, this->ppcell->vkb.c, &max_npw, ps1_minus2.c, &n_npwx,
           &ModuleBase::ONE, vpsi+max_npw * n_npwx, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &negfactor, this->ppcell->vkb_minus2.c, &max_npw, ps1.c, &n_npwx,
               &ModuleBase::ONE, vpsi+max_npw * n_npwx, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &factor, this->ppcell->vkb.c, &max_npw, ps1_plus2.c, &n_npwx,
               &ModuleBase::ONE, vpsi+max_npw * n_npwx, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &factor, this->ppcell->vkb_plus2.c, &max_npw, ps1.c, &n_npwx,
               &ModuleBase::ONE, vpsi+max_npw * n_npwx, &max_npw);

        zgemm_(&transN, &transT, &npw, &npm, &nkb,
           &negfactor, this->ppcell->vkb.c, &max_npw, ps1_minus3.c, &n_npwx,
           &ModuleBase::ONE, vpsi+max_npw * n_npwx*2, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &negfactor, this->ppcell->vkb_minus3.c, &max_npw, ps1.c, &n_npwx,
               &ModuleBase::ONE, vpsi+max_npw * n_npwx*2, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &factor, this->ppcell->vkb.c, &max_npw, ps1_plus3.c, &n_npwx,
               &ModuleBase::ONE, vpsi+max_npw * n_npwx*2, &max_npw);
        zgemm_(&transN, &transT, &npw, &npm, &nkb,
               &factor, this->ppcell->vkb_plus3.c, &max_npw, ps1.c, &n_npwx,
               &ModuleBase::ONE, vpsi+max_npw * n_npwx*2, &max_npw);
        
    }
    // for(int id = 0 ; id < 3 ; ++id)
    // {
    //     for(int ib = 0 ; ib < n_npwx ; ++ib)
    //     {
    //         const int i0 = ib * max_npw;
    //         for(int i = 0 ; i < npw ; ++i)
    //         {
    //             vpsi2[ i0 + i + id * max_npw * n_npwx] /= (2 * thr * wfcpw->tpiba); 
    //             if(abs(vpsi2[i + i0+ id * max_npw * n_npwx]- vpsi[i + i0 + id * max_npw * n_npwx] ) > 1e-6)
    //                 // ModuleBase::WARNING_QUIT("velocity_pw.cpp","Wrong");
    //             cout<<id<<" "<<ib<<" "<<i<<" "<<vpsi2[i + i0+ id * max_npw * n_npwx] <<" "<<vpsi[i + i0 + id * max_npw * n_npwx] 
    //                 <<" "<< vpsi[i + i0 + id * max_npw * n_npwx] / vpsi2[i + i0+ id * max_npw * n_npwx]<<endl;
    //         }
    //     }
    // }

    // delete[] vpsi2;
    ModuleBase::timer::tick("Operator", "Velocity");
    return;
}


}