#ifndef VELOCITY_PW_H
#define VELOCITY_PW_H
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_cell/unitcell.h"
#include "module_hamilt_pw/hamilt_pwdft/VNL_in_pw.h"
#include "operator_pw.h"
namespace hamilt
{

// velocity operator mv = im/\hbar * [H,r] =  p + im/\hbar [V_NL, r]
class Velocity
{
  public:
    Velocity(const ModulePW::PW_Basis_K* wfcpw_in,
             const int* isk_in,
             pseudopot_cell_vnl* ppcell_in,
             const UnitCell* ucell_in,
             const bool nonlocal_in = true);

    ~Velocity()
    {
        delete[] f_gx;
        delete[] f_gy;
        delete[] f_gz;
        delete[] f_vkb;
        delete[] f_gradvkb;
        delete[] f_deeq;
    };

    void init(const int ik_in, const bool init_float = false);

    /**
     * @brief calculate \hat{v}|\psi>
     *
     * @param psi_in Psi class which contains some information
     * @param n_npwx i = 1 : n_npwx
     * @param tmpsi_in |\psi_i>    size: n_npwx*npwx
     * @param tmhpsi \hat{v}|\psi> size: 3*n_npwx*npwx
     * @param add true : tmhpsi = tmhpsi + v|\psi>  false: tmhpsi = v|\psi>
     *
     */
    void act(const psi::Psi<std::complex<double>>* psi_in,
             const int n_npwx,
             const std::complex<double>* tmpsi_in,
             std::complex<double>* tmhpsi,
             const bool add = false) const;
    void act(const psi::Psi<std::complex<float>>* psi_in,
             const int n_npwx,
             const std::complex<float>* tmpsi_in,
             std::complex<float>* tmhpsi,
             const bool add = false) const;

    bool nonlocal = true;

  private:
    const ModulePW::PW_Basis_K* wfcpw = nullptr;

    const int* isk = nullptr;

    pseudopot_cell_vnl* ppcell = nullptr;

    const UnitCell* ucell = nullptr;

    int ik;

    double tpiba;

  private:
    // for float
    float* f_gx = nullptr;
    float* f_gy = nullptr;
    float* f_gz = nullptr;
    std::complex<float>* f_vkb = nullptr;
    std::complex<float>* f_gradvkb = nullptr;
    float* f_deeq = nullptr;
};
} // namespace hamilt
#endif