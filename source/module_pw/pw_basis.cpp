#include "pw_basis.h"
#include "../module_base/mymath.h"
#include <iostream>
#include "../module_base/timer.h"
#include "../module_base/global_function.h"


namespace ModulePW
{

PW_Basis::PW_Basis()
{
    ig2isz = nullptr;
    istot2bigixy = nullptr;   
    ixy2istot = nullptr;
    is2ixy = nullptr;
    ixy2ip = nullptr;
    startnsz_per =nullptr;
    nstnz_per = nullptr;
    nst_per = nullptr;
    gdirect = nullptr;		
    gcar = nullptr; 
    gg = nullptr;
    startz = nullptr;
    numz = nullptr; 
    numg = nullptr;
	startg = nullptr;
	startr = nullptr;
	numr = nullptr;
    ig2igg = nullptr;
    gg_uniq = nullptr;
    ig_gge0 = -1;
    poolnproc = 1;
    poolrank = 0;
    npw = 0;
    npwtot = 0;
    ngg = 0;
}

PW_Basis:: ~PW_Basis()
{
    if(ig2isz != nullptr)           delete[] ig2isz;
    if(istot2bigixy != nullptr)     delete[] istot2bigixy;
    if(ixy2istot != nullptr)        delete[] ixy2istot;
    if(is2ixy != nullptr)           delete[] is2ixy;
    if(ixy2ip != nullptr)           delete[] ixy2ip;
    if(startnsz_per != nullptr)     delete[] startnsz_per;
    if(nstnz_per != nullptr)        delete[] nstnz_per;
    if(nst_per != nullptr)          delete[] nst_per;
    if(gdirect != nullptr)          delete[] gdirect;
    if(gcar != nullptr)             delete[] gcar;
    if(gg != nullptr)               delete[] gg;
    if(startz != nullptr)           delete[] startz;
    if(numz != nullptr)             delete[] numz;
    if(numg != nullptr)             delete[] numg;
    if(numr != nullptr)             delete[] numr;
    if(startg != nullptr)           delete[] startg;
    if(startr != nullptr)           delete[] startr;
    if(ig2igg != nullptr)           delete[] ig2igg;
    if(gg_uniq != nullptr)          delete[] gg_uniq;
}

/// 
/// distribute plane wave basis and real-space grids to different processors
/// set up maps for fft and create arrays for MPI_Alltoall
/// set up ffts
///
void PW_Basis::setuptransform()
{
    this->distribute_r();
    this->distribute_g();
    this->getstartgr();
    this->ft.clear();
    this->ft.initfft(this->nx,this->bigny,this->nz,this->liy,this->riy,this->nst,this->nplane,this->poolnproc,this->gamma_only);
    this->ft.setupFFT();
}

void PW_Basis::getstartgr()
{
    if(this->gamma_only)    this->nmaxgr = ( this->npw > (this->nrxx+1)/2 ) ? this->npw : (this->nrxx+1)/2;
    else                    this->nmaxgr = ( this->npw > this->nrxx ) ? this->npw : this->nrxx;
    this->nmaxgr = (this->nz * this->nst > this->bignxy * nplane) ? this->nz * this->nst : this->bignxy * nplane;
    
    //---------------------------------------------
	// sum : starting plane of FFT box.
	//---------------------------------------------
    if(this->numg!=nullptr) delete[] this->numg; this->numg = new int[poolnproc];
	if(this->startg!=nullptr) delete[] this->startg; this->startg = new int[poolnproc];
	if(this->startr!=nullptr) delete[] this->startr; this->startr = new int[poolnproc];
	if(this->numr!=nullptr) delete[] this->numr; this->numr = new int[poolnproc];

	// Each processor has a set of full sticks,
	// 'rank_use' processor send a piece(npps[ip]) of these sticks(nst_per[rank_use])
	// to all the other processors in this pool
	for (int ip = 0;ip < poolnproc; ++ip) this->numg[ip] = this->nst_per[poolrank] * this->numz[ip];


	// Each processor in a pool send a piece of each stick(nst_per[ip]) to
	// other processors in this pool
	// rank_use processor receive datas in npps[rank_p] planes.
	for (int ip = 0;ip < poolnproc; ++ip) this->numr[ip] = this->nst_per[ip] * this->numz[poolrank];


	// startg record the starting 'numg' position in each processor.
	this->startg[0] = 0;
	for (int ip = 1;ip < poolnproc; ++ip) this->startg[ip] = this->startg[ip-1] + this->numg[ip-1];


	// startr record the starting 'numr' position
	this->startr[0] = 0;
	for (int ip = 1;ip < poolnproc; ++ip) this->startr[ip] = this->startr[ip-1] + this->numr[ip-1];
    return;
}

///
/// Collect planewaves on current core, and construct gg, gdirect, gcar according to ig2isz and is2ixy.
/// known: ig2isz, is2ixy
/// output: gg, gdirect, gcar
/// 
void PW_Basis::collect_local_pw()
{
   if(this->gg!=nullptr) delete[] this->gg; this->gg = new double[this->npw];
   if(this->gdirect!=nullptr) delete[] this->gdirect; this->gdirect = new ModuleBase::Vector3<double>[this->npw];
   if(this->gcar!=nullptr) delete[] this->gcar; this->gcar = new ModuleBase::Vector3<double>[this->npw];

    ModuleBase::Vector3<double> f;
    for(int ig = 0 ; ig < this-> npw ; ++ig)
    {
        int isz = this->ig2isz[ig];
        int iz = isz % this->nz;
        int is = isz / this->nz;
        int ixy = this->is2ixy[is];
        int ix = ixy / this->ny;
        int iy = ixy % this->ny;
        if (ix >= int(this->nx/2) + 1) ix -= this->nx;
        if (iy >= int(this->bigny/2) + 1) iy -= this->bigny;
        if (iz >= int(this->nz/2) + 1) iz -= this->nz;
        f.x = ix;
        f.y = iy;
        f.z = iz;
        this->gg[ig] = f * (this->GGT * f);
        this->gdirect[ig] = f;
        this->gcar[ig] = f * this->G;
        if(this->gg[ig] < 1e-8) this->ig_gge0 = ig;
    }
    return;
}
void PW_Basis::collect_uniqgg()
{
    if(this->ig2igg!=nullptr) delete[] this->ig2igg; this->ig2igg = new int [this->npw];
    int *sortindex = new int [this->npw];
    double *tmpgg = new double [this->npw];
    double *tmpgg2 = new double [this->npw];
    ModuleBase::Vector3<double> f;
    for(int ig = 0 ; ig < this-> npw ; ++ig)
    {
        int isz = this->ig2isz[ig];
        int iz = isz % this->nz;
        int is = isz / this->nz;
        int ixy = this->is2ixy[is];
        int ix = ixy / this->ny;
        int iy = ixy % this->ny;
        if (ix >= int(this->nx/2) + 1) ix -= this->nx;
        if (iy >= int(this->bigny/2) + 1) iy -= this->bigny;
        if (iz >= int(this->nz/2) + 1) iz -= this->nz;
        f.x = ix;
        f.y = iy;
        f.z = iz;
        tmpgg[ig] = f * (this->GGT * f);
        if(tmpgg[ig] < 1e-8) this->ig_gge0 = ig;
    }

    ModuleBase::GlobalFunc::ZEROS(sortindex, this->npw);
    ModuleBase::heapsort(this->npw, tmpgg, sortindex);
   

    int igg = 0;
    this->ig2igg[sortindex[0]] = 0;
    tmpgg2[0] = tmpgg[0];
    double avg_gg = tmpgg2[igg];
    int avg_n = 1;
    for (int ig = 1; ig < this->npw; ++ig)
    {
        if (std::abs(tmpgg[ig] - tmpgg2[igg]) > 1.0e-8)
        {
            tmpgg2[igg] = avg_gg / double(avg_n) ;
            ++igg;
            tmpgg2[igg] = tmpgg[ig];
            avg_gg = tmpgg2[igg];
            avg_n = 1;   
        }
        else
        {
            avg_n++;
            avg_gg += tmpgg[ig];
        }
        this->ig2igg[sortindex[ig]] = igg;
        if(ig == this->npw)
        {
            tmpgg2[igg] = avg_gg / double(avg_n) ;
        }
    }
    this->ngg = igg + 1;
    if(this->gg_uniq!=nullptr) delete[] this->gg_uniq; this->gg_uniq = new double [this->ngg];
    for(int igg = 0 ; igg < this->ngg ; ++igg)
    {
            gg_uniq[igg] = tmpgg2[igg];
    }
    delete[] sortindex;
    delete[] tmpgg;
    delete[] tmpgg2;
}

// //
// // Collect total planewaves, store moduli to gg_global, direct coordinates to gdirect_global, and Cartesian coordinates to gcar_global,
// // planewaves are stored in the order of modulus increasing.
// // known: nx, ny, nz, ggcut
// // output: gg_global, gdirect_global, gcar_global
// // 
// void PW_Basis::collect_tot_pw(double* gg_global, ModuleBase::Vector3<double> *gdirect_global, ModuleBase::Vector3<double> *gcar_global)
// {
//     int tot_npw = 0;
//     int ibox[3] = {0, 0, 0};                            // an auxiliary vector, determine the boundary of the scanning area.
//     ibox[0] = int(this->nx / 2) + 1;                    // scan x from -ibox[0] to ibox[0].
//     ibox[1] = int(this->ny / 2) + 1;                    // scan y from -ibox[1] to ibox[1].
//     ibox[2] = int(this->nz / 2) + 1;                    // scan z from -ibox[2] to ibox[2].

//     ModuleBase::Vector3<double> f;

//     int ix_start = -ibox[0]; // determine the scaning area along x-direct, if gamma-only, only positive axis is used.
//     int ix_end = ibox[0];
//     if (this->gamma_only)
//     {
//         ix_start = 0;
//         ix_end = this->nx;
//     }

//     // count the number of planewaves
//     for (int iy = -ibox[1]; iy <= ibox[1]; ++iy)
//     {
//         for (int ix = ix_start; ix <= ix_end; ++ix)
//         {
//             // we shift all sticks to the first quadrant in x-y plane here.
//             // (ix, iy, iz) is the direct coordinates of planewaves.
//             // x and y is the coordinates of shifted sticks in x-y plane.
//             // for example, if nx = ny = 10, we will shift the stick on (-1, 2) to (9, 2),
//             // so that its index in st_length and st_bottom is 9 + 10 * 2 = 29.
//             int x = ix;
//             int y = iy;
//             if (x < 0) x += this->nx;
//             if (y < 0) y += this->ny;
//             int index = y * this->nx + x;

//             int length = 0; // number of planewave in stick (x, y).
//             for (int iz = -ibox[2]; iz <= ibox[2]; ++iz)
//             {
//                 f.x = ix;
//                 f.y = iy;
//                 f.z = iz;
//                 double modulus = f * (this->GGT * f);
//                 if (modulus <= this->ggecut) ++tot_npw;
//             }
//         }
//     }

//     // find all the planewaves
//     if (gg_global != NULL) delete[] gg_global;
//     if (gdirect_global != NULL) delete[] gdirect_global;
//     if (gcar_global != NULL) delete[] gcar_global;
//     gg_global = new double[tot_npw];
//     gdirect_global = new ModuleBase::Vector3<double>[tot_npw];
//     gcar_global = new ModuleBase::Vector3<double>[tot_npw];
//     ModuleBase::Vector3<double> *temp_gdirect = new ModuleBase::Vector3<double>[tot_npw]; // direct coordinates of all planewaves, in the order of (x * ny * nz + y * nx + z).
//     int ig = 0; // index of planewave.
//     for (int iy = -ibox[1]; iy <= ibox[1]; ++iy)
//     {
//         for (int ix = ix_start; ix <= ix_end; ++ix)
//         {
//             for (int iz = -ibox[2]; iz <= ibox[2]; ++iz)
//             {
//                 f.x = ix;
//                 f.y = iy;
//                 f.z = iz;
//                 double modulus = f * (GGT * f);
//                 if (modulus <= ggecut)
//                 {
//                     gg_global[ig] = modulus;
//                     temp_gdirect[ig] = f;
//                     ig++; 
//                 }
//             }
//         }
//     }
//     assert(ig == tot_npw);

//     std::cout<<"collect pw and sticks done\n";
//     for (int ig = 0; ig < tot_npw; ++ig)
//     {
//         std::cout << gg_global[ig] << std::setw(4);
//     }
//     std::cout << '\n';

//     // Rearrange gg_global and gdirect in the order of modulus decreasing, and sort st_* from longest to shortest.
//     int *gg_sorted_index = new int[tot_npw]; // indexs of planewaves in the order of modulus increasing.
//     gg_sorted_index[0] = 0;
//     ModuleBase::heapsort(tot_npw, gg_global, gg_sorted_index); // sort gg_global in the order of modulus decreasing.
//     for (int igtot = 0; igtot < tot_npw; ++igtot)
//     {
//         gdirect_global[igtot] = temp_gdirect[gg_sorted_index[igtot]]; // rearrange gdirect_global in the same order of gg_global.
//         gcar_global[igtot] = gdirect_global[igtot] * this->G;
//     }
//     std::cout<<"sort pw done\n";

//     delete[] temp_gdirect;
//     delete[] gg_sorted_index;
// }

 }