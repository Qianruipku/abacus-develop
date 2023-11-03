#ifndef STO_HCHI_H
#define STO_HCHI_H
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_cell/klist.h"

//-----------------------------------------------------
// h * chi
// chi: stochastic wave functions
//
// h: the normalized Hamiltonian matrix, which equals to (H - E\bar) / \DeltaE
// where H is decomposed into the electron kinetic, effective potential V(r),
// and the non-local pseudopotentials.
// The effective potential = Local pseudopotential +
// Hartree potential + Exchange-correlation potential
//------------------------------------------------------
class Stochastic_hchi
{

  public:
    // constructor and deconstructor
    Stochastic_hchi();
    ~Stochastic_hchi();

    void init(ModulePW::PW_Basis_K* wfc_basis, K_Vectors* pkv);

    // update hchi for ik
    void updateik_float(const int& ik);

    double Emin;
    double Emax;

    /**
     * @brief calculate H|\chi>
     * 
     * @param wfin input stochastic wave functions
     * @param wfout output stochastic wave functions
     * @param m number of stochastic wave functions
     */
    void hchi(std::complex<double>* wfin, std::complex<double>* wfout, const int m = 1);

    void hchi_norm(std::complex<double>* wfin, std::complex<double>* wfout, const int m = 1);

    void hchi(std::complex<float>* wfin, std::complex<float>* wfout, const int m = 1);

    void hchi_norm(std::complex<float>* wfin, std::complex<float>* wfout, const int m = 1);

  public:
    int current_ik = 0;
    ModulePW::PW_Basis_K* wfcpw = nullptr;
    K_Vectors* pkv = nullptr;

  private:
    // for float
    float* f_king = nullptr;
    float* f_veff = nullptr;
    std::complex<float>* f_vkb = nullptr;
    float* f_deeq = nullptr;
};

#endif // Eelectrons_hchi
