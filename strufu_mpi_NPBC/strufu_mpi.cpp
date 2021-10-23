/*

  strufu_mpi.cpp

  Computes structure functions of
  grid variables from FLASH output files

  By Christoph Federrath, 2013-2021

*/

#include "mpi.h" /// MPI lib
#include "stdlib.h"
#include <assert.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <limits> /// numeric limits
#include "../Libs/FlashGG.h" /// Flash General Grid class

// constants
#define NDIM 3
using namespace std;
enum {X, Y, Z};
static const bool Debug = false;
static const int MAX_NUM_BINS = 10048;
static const double k_b = 1.380649e-16;
static const double m_p = 1.67262192369e-24;
static const double mu = 2.0;

// MPI stuff
int MyPE = 0, NPE = 1;

// some global stuff (inputs)
FlashGG gg; // global FlashGG object
char GridType;
string inputfile = "";
double n_samples = 0.0;
int ncells_pseudo_blocks = 0;
bool ncells_pseudo_blocks_set = false;
string OutputPath = "./";

// for output
vector<string> OutputFileHeader;
vector< vector<double> > WriteOutTable;

// forward functions
float * ReadBlock(const int block, const string datasetname);
void ComputeStructureFunctions(void);
void WriteOutAnalysedData(const string OutputFilename);
int ParseInputs(const vector<string> Argument);
void HelpMe(void);


/// --------
///   MAIN
/// --------
int main(int argc, char * argv[])
{

    /*
    // random number test
    int n = 10000;
    double x[n], y[n], z[n];
    mt19937 seed1(0);
    for (int i = 0; i < n; i++) x[i] = random_number(seed1);
    mt19937 seed2(10);
    for (int i = 0; i < n; i++) y[i] = random_number(seed2);
    mt19937 seed3(20);
    for (int i = 0; i < n; i++) z[i] = random_number(seed3);
    for (int i = 0; i < n; i++) cout << x[i] << " " << y[i] << " " << z[i] << " " << endl;
    exit(0);
    */

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &NPE);
    MPI_Comm_rank(MPI_COMM_WORLD, &MyPE);
    long starttime = time(NULL);

    if (MyPE==0) cout<<"=== strufu_mpi === using MPI num procs: "<<NPE<<endl;

    /// Parse inputs
    vector<string> Arguments(argc);
    for (int i = 0; i < argc; i++) Arguments[i] = static_cast<string>(argv[i]);
    if (ParseInputs(Arguments) == -1)
    {
        if (MyPE==0) cout << endl << "Error in ParseInputs(). Exiting." << endl;
        HelpMe();
        MPI_Finalize(); return 0;
    }

    /// read data, compute SF, and output
    ComputeStructureFunctions();

    /// print out wallclock time used
    long endtime = time(NULL);
    long duration = endtime-starttime;
    long duration_red = 0;
    if (Debug) cout << "["<<MyPE<<"] ****************** Local time to finish = "<<duration<<"s ******************" << endl;
    MPI_Allreduce(&duration, &duration_red, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    duration = duration_red;
    if (MyPE==0) cout << "****************** Global time to finish = "<<duration<<"s ******************" << endl;

    MPI_Finalize();
    return 0;

} // end main


// function call to gg.ReadBlockVar ot gg.ReadBlockVar_PB, depending on whether grid is uniform, extracted, or AMR
float * ReadBlock(const int block, const string datasetname)
{
    float *tmp = 0;
    if (GridType == 'U' || GridType == 'E')
        tmp = gg.ReadBlockVar_PB(block, datasetname);
    else
        tmp = gg.ReadBlockVar(block, datasetname);
    return tmp;
}


/** ---------------- ComputeStructureFunctions -----------------------
 ** computes longitudinal and transverse structure functions
 ** up to the 10th order (this assumes a cubic box !)
 ** ------------------------------------------------------------------ */
void ComputeStructureFunctions(void)
{
    if (MyPE==0 && Debug) cout<<"ComputeStructureFunctions: entering."<<endl;

    /// FLASH file meta data
    gg = FlashGG(inputfile);
    GridType = gg.GetGridType();
    vector< vector<double> > MinMaxDomain = gg.GetMinMaxDomain();
    vector<double> L = gg.GetL();
    vector<double> LHalf(3); for (int dir=X; dir<=Z; dir++) LHalf[dir] = L[dir]/2.;
    vector<double> Dmin = gg.GetDmin();
    vector<double> Dmax = gg.GetDmax();
    vector< vector<double> > D = gg.GetDblock();
    vector<int> N = gg.GetN();
    assert (N[X] == N[Y] && N[X] == N[Z]); // make sure this is a cube where N[X] = N[Y] = N[Z]
    vector<int> LeafBlocks = gg.GetLeafBlocks();
    int NBLK = LeafBlocks.size();
    vector<int> NB = gg.GetNumCellsInBlock();
    vector< vector <vector<double> > > BB = gg.GetBoundingBox();
    vector< vector<double> > LBlock = gg.GetLBlock();

    if (GridType == 'U' || GridType == 'E') { // uniform or extracted grid
        // automatically determine a good value for ncells_pb, if not set by user
        if (!ncells_pseudo_blocks_set) {
            // get integer multiples of N
            vector<int> integer_multiples(0);
            for (int n = 1; n <= N[X]; n++)
                if (N[X] % n == 0) integer_multiples.push_back(n);
            // print possible value of ncells_pb to screen and take a good one out (one for which N/multiple <~ 8)
            bool done_setting = false;
            if (MyPE==0) cout<<"ComputeStructureFunctions: Possible values for -ncells_pb: ";
            for (int i = 0; i < integer_multiples.size(); i++) {
                if (MyPE==0) cout<<integer_multiples[i]<<" ";
                if ((N[X]/integer_multiples[i] <= 8) && (not done_setting)) {
                    ncells_pseudo_blocks = integer_multiples[i];
                    done_setting = true;
                }
            }
            if (MyPE==0) cout<<endl<<"ComputeStructureFunctions: Automatically selected value -ncells_pb "<<ncells_pseudo_blocks<<endl;
        }
        // setup pseudo blocks
        vector<int> NB_PB(3);
        NB_PB[X] = ncells_pseudo_blocks;
        NB_PB[Y] = ncells_pseudo_blocks;
        NB_PB[Z] = ncells_pseudo_blocks;
        if (MyPE == 0) cout << "Using pseudo blocks with "<<ncells_pseudo_blocks<<" (cubed) cells."<< endl;
        gg.SetupPseudoBlocks(NB_PB);
        // specific to pseudo block setup
        NBLK = gg.GetNumBlocks_PB();
        NB = gg.GetNumCellsInBlock_PB();
        BB = gg.GetBoundingBox_PB();
        LBlock = gg.GetLBlock_PB();
        LeafBlocks = vector<int>(NBLK); for (int ib=0; ib<NBLK; ib++) { LeafBlocks[ib] = ib; }
        D = vector< vector<double> >(NBLK, Dmin); // overwrite D with UG or Extracted Grid version
    }

    double LBlockMin = 1e99;
    for (int ib=0; ib<NBLK; ib++) {
        int ibl = LeafBlocks[ib];
        if (LBlock[ibl][X] < LBlockMin) LBlockMin = LBlock[ibl][X];
    }

    // set more meta data and print info
    long NBXYZ = NB[X]*NB[Y]*NB[Z];
    if (MyPE==0) gg.PrintInfo();

    /// decompose domain in blocks
    vector<int> MyBlocks = gg.GetMyBlocks(MyPE, NPE);

    if (Debug) {
        cout<<" ["<<MyPE<<"] MyBlocks =";
        for (unsigned int ib=0; ib<MyBlocks.size(); ib++)
            cout<<" "<<MyBlocks[ib];
        cout<<endl;
    }

    // set the number of samples if the user has not supplied a number
    if (n_samples == 0.0) {
        n_samples = (10.0*NBLK)*(10.0*NBLK);
        if (MyPE==0) {
            cout<<"ComputeStructureFunctions: Number of samples was not set; calculating a guess of -n "<<n_samples<<"."<<endl;
            cout<<"  Suggest to increase manually with the -n option, and check for statistical convergence."<<endl;
        }
    }
    if (MyPE==0) cout << "ComputeStructureFunctions: Using about "<<n_samples<<" total samples."<<endl;
    if (MyPE==0) cout << "ComputeStructureFunctions: === Start looping ==="<<endl;

    /// initialization
    const int    MaxStructureFunctionOrder = 10;
    const int    NumTypes = 1; /// v, sqrt(rho)*v, rho^(1/3)*v, rho*v, rho, rhotohalf, ln(rho), gb11 Flux, gb11 S
    const double onethird = static_cast<double>(1.0/3.0);

    /// construct bins grid
    int    NumberOfBins = 0;
    double grid     [MAX_NUM_BINS]; grid     [0] = 0.0;
    double grid_stag[MAX_NUM_BINS]; grid_stag[0] = Dmax[X]/2.;
    const double MaxLength = 0.5*L[X]; /// max length between two different points in a cubic box
    double length = Dmax[X];
    while (grid[NumberOfBins] < 0.5*sqrt(3.0)*L[X]) /// max distance
    {
        NumberOfBins++;
        grid     [NumberOfBins] = length;
        grid_stag[NumberOfBins] = length + Dmax[X]/2.;
        length += Dmax[X];
        if (NumberOfBins > MAX_NUM_BINS) {
            if (MyPE==0) cout << "ERROR. NumberOfBins exceeds MaximumNumberofBins!" << endl;
            MPI_Finalize();
        }
    }
    NumberOfBins++;

    /// initialize main buffers
    const double numeric_epsilon = 10*numeric_limits<double>::epsilon();
    const int    NumBufferIndices = NumTypes*MaxStructureFunctionOrder*NumberOfBins;

    /// initialize additional buffers
    double     *buf1_bin_counter_long           = new double    [NumBufferIndices];
    double     *buf1_bin_counter_trsv           = new double    [NumBufferIndices];
    double     *buf1_binsum_long                = new double    [NumBufferIndices];
    double     *buf1_binsum_trsv                = new double    [NumBufferIndices];
    long       *buf1_numeric_error_counter_long = new long      [NumBufferIndices];
    long       *buf1_numeric_error_counter_trsv = new long      [NumBufferIndices];
    double     *buf2_bin_counter_long           = new double    [NumBufferIndices];
    double     *buf2_bin_counter_trsv           = new double    [NumBufferIndices];
    double     *buf2_binsum_long                = new double    [NumBufferIndices];
    double     *buf2_binsum_trsv                = new double    [NumBufferIndices];
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        buf1_bin_counter_long          [BufIndex] = 0;
        buf1_bin_counter_trsv          [BufIndex] = 0;
        buf1_binsum_long               [BufIndex] = numeric_epsilon;
        buf1_binsum_trsv               [BufIndex] = numeric_epsilon;
        buf1_numeric_error_counter_long[BufIndex] = 0;
        buf1_numeric_error_counter_trsv[BufIndex] = 0;
        buf2_bin_counter_long          [BufIndex] = 0;
        buf2_bin_counter_trsv          [BufIndex] = 0;
        buf2_binsum_long               [BufIndex] = numeric_epsilon;
        buf2_binsum_trsv               [BufIndex] = numeric_epsilon;
    }

    double DX[NumTypes] = {0}; double DY[NumTypes] = {0}; double DZ[NumTypes] = {0};
    double incr_long[NumTypes] = {0}; double incr_trsv[NumTypes] = {0};
    double incr_long_pow[NumTypes] = {0}; double incr_trsv_pow[NumTypes] = {0};

    // for random numbers (Mersenne-Twister)
    uniform_real_distribution<double> random_number(0.0, 1.0);
    // different random seed for each PE
    mt19937 seed(MyPE);

    /// loop over my blocks (b1)
    for (unsigned int ib1=0; ib1<MyBlocks.size(); ib1++)
    {
        int b1 = MyBlocks[ib1];
        if (MyPE==0) cout<<" ["<<setw(6)<<MyPE<<"] working on block1 = "<<setw(6)<<ib1+1<<" of "<<MyBlocks.size()<<endl;

        // block center 1
        vector<double> bc1(3);
        bc1[X] = (BB[b1][X][0]+BB[b1][X][1])/2.;
        bc1[Y] = (BB[b1][Y][0]+BB[b1][Y][1])/2.;
        bc1[Z] = (BB[b1][Z][0]+BB[b1][Z][1])/2.;
        double block_diagonal = sqrt( (BB[b1][X][1]-BB[b1][X][0])*(BB[b1][X][1]-BB[b1][X][0]) +
                                      (BB[b1][Y][1]-BB[b1][Y][0])*(BB[b1][Y][1]-BB[b1][Y][0]) +
                                      (BB[b1][Z][1]-BB[b1][Z][0])*(BB[b1][Z][1]-BB[b1][Z][0]) );

        /// read block data
        if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block1..."<<endl;
        float *dens1 = 0; // ReadBlock(b1, "dens");
        if (MyPE==0 && Debug && dens1) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 dens done."<<endl;
        float *temp1 = 0; // ReadBlock(b1, "temp");
        if (MyPE==0 && Debug && temp1) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 temp done."<<endl;
        float *velx1 = ReadBlock(b1, "velx");
        if (MyPE==0 && Debug && velx1) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 velx done."<<endl;
        float *vely1 = ReadBlock(b1, "vely");
        if (MyPE==0 && Debug && vely1) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 vely done."<<endl;
        float *velz1 = ReadBlock(b1, "velz");
        if (MyPE==0 && Debug && velz1) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 velz done."<<endl;
        float *divv1 = 0; // ReadBlock(b1, "divv");
        if (MyPE==0 && Debug && divv1) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 divv done."<<endl;
        if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block1 done."<<endl;

        /// loop over all blocks (b2)
        bool printed_progress_1_b2 = false, printed_progress_10_b2 = false, printed_progress_100_b2 = false;
        for (unsigned int ib2=0; ib2<NBLK; ib2++)
        {
            int b2 = LeafBlocks[ib2];

            // write progress
            double percent_done = (double)(b2+1)/NBLK*100;
            bool print_progress = false;
            if (percent_done >    1.0 && !printed_progress_1_b2  ) {print_progress=true; printed_progress_1_b2  =true;}
            if (percent_done >   10.0 && !printed_progress_10_b2 ) {print_progress=true; printed_progress_10_b2 =true;}
            if (percent_done == 100.0 && !printed_progress_100_b2) {print_progress=true; printed_progress_100_b2=true;}
            if (print_progress && MyPE==0) cout<<"   ..."<<percent_done<<"% done..."<<endl;

            // total volume (1 for UG or Extracted grid; or sum of cell1 and cell2 vol for AMR grid)
            double TotVol = 1.0;
            if (GridType == 'A') TotVol = D[b1][X]*D[b1][Y]*D[b1][Z] + D[b2][X]*D[b2][Y]*D[b2][Z];

            // block center 2
            vector<double> bc2(3);
            bc2[X] = (BB[b2][X][0]+BB[b2][X][1])/2.;
            bc2[Y] = (BB[b2][Y][0]+BB[b2][Y][1])/2.;
            bc2[Z] = (BB[b2][Z][0]+BB[b2][Z][1])/2.;

            // distance between block center 1 and 2
            vector<double> dbc(3);
            dbc[X] = bc2[X]-bc1[X]; 
            dbc[Y] = bc2[Y]-bc1[Y]; 
            dbc[Z] = bc2[Z]-bc1[Z]; 

            double block_distance = sqrt(dbc[X]*dbc[X]+dbc[Y]*dbc[Y]+dbc[Z]*dbc[Z]);

            if (Debug) cout << block_distance << " " << dbc[X] << " " << dbc[Y] << " " << dbc[Z] << endl;

            /// loop parameters for samples in block 1
            long n_samples1 = max(1.0,round(sqrt((double)(n_samples))/(double)(NBLK)));
            if (n_samples1 < 10 && MyPE == 0) cout << "Warning: n_samples1 = "<<n_samples1<<
                " < 10 !!! Consider increasing n samples; suggesting n > " << (10.0*NBLK)*(10.0*NBLK) << endl;

            /// loop parameters for samples in block 2
            long n_samples2 = min((double)(n_samples1),(double)NBXYZ);
            if (Debug) cout << "n_samples2 = " << n_samples2 << endl;

            double rmin = LBlockMin, rmax = MaxLength;
            double n_samples2_norm = (double)(n_samples2)/(rmax/rmin-0.9);
            long n_samples2_block = n_samples2_norm*pow(block_distance/rmax,-2.0);
            if (block_distance == 0) n_samples2_block = n_samples2_norm*n_samples2;
            if (Debug) {
                cout << "rmin = " << rmin << endl;
                cout << "rmax = " << rmax << endl;
                cout << "(rmax/rmin-1.0) = " << (rmax/rmin-1.0) << endl;
                cout << "n_samples2_norm = " << n_samples2_norm << endl;
                cout << "n_samples2_block = " << n_samples2_block << endl;
            }

            double NBS = (double)n_samples2_block; if (NBS==0) NBS = 1.0;
            int incr = max((double)(NBXYZ)/NBS,1.0);
            if (incr == 1 && MyPE == 0) cout << "Warning: looping over ALL cells in block2." << endl;
            if (n_samples1*n_samples2_block > 1e9)
            {
                if (MyPE == 0) cout << "Warning: n_samples1*n_samples2_block > 1e9. Resetting to 1e9." << endl;
                incr = (int)max((double)(NBXYZ)/(1e9/n_samples1),1.);
            }

            if (Debug) cout << block_distance << " " << MaxLength+block_diagonal << " " << n_samples2_block << " " << incr << " " << NBXYZ << endl;

            /// continue with next block if too far away or no samples
            if ((block_distance > MaxLength+block_diagonal) || (n_samples2_block < 1) || incr >= NBXYZ) continue;

            /// read block data
            if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block2..."<<endl;
            float *dens2 = 0; // ReadBlock(b2, "dens");
            if (MyPE==0 && Debug && dens2) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 dens done."<<endl;
            float *temp2 = 0; // ReadBlock(b2, "temp");
            if (MyPE==0 && Debug && temp2) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 temp done."<<endl;
            float *velx2 = ReadBlock(b2, "velx");
            if (MyPE==0 && Debug && velx2) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 velx done."<<endl;
            float *vely2 = ReadBlock(b2, "vely");
            if (MyPE==0 && Debug && vely2) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 vely done."<<endl;
            float *velz2 = ReadBlock(b2, "velz");
            if (MyPE==0 && Debug && velz2) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 velz done."<<endl;
            float *divv2 = 0; // ReadBlock(b2, "divv");
            if (MyPE==0 && Debug && divv2) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 divv done."<<endl;
            if (MyPE==0 && Debug) cout<<" ["<<setw(6)<<MyPE<<"] reading block2 done. block1 = "<<setw(6)<<ib1+1<<" of "<<MyBlocks.size()<<
                         "; block2 count on root processor = "<<setw(6)<<b2+1<<" of "<<NBLK<<" now computing SFs..."<<endl;

            if (Debug && MyPE==0) cout<<"block_distance="<<setw(16)<<block_distance<<" n_samples1="<<setw(16)<<n_samples1
                             <<" n_samples2_block="<<setw(16)<<(int)((double)NBXYZ/(double)incr)<<endl;

            for (long n1=0; n1<n_samples1; n1++)
            {
                int i1(NB[X]*(random_number(seed)));
                int j1(NB[Y]*(random_number(seed)));
                int k1(NB[Z]*(random_number(seed)));

                long cellindex1 = k1*NB[Y]*NB[X] + j1*NB[X] + i1;
                vector<double> cc1;
                if (GridType == 'U' || GridType == 'E')
                    cc1 = gg.CellCenter_PB(b1, i1, j1, k1);
                else // AMR
                    cc1 = gg.CellCenter(b1, i1, j1, k1);
                if (Debug) cout<<">>> cc1 = "<<cc1[X]<<" "<<cc1[Y]<<" "<<cc1[Z]<<endl;

                for (long cellindex2=0; cellindex2<NBXYZ; cellindex2+=incr)
                {
                    vector<double> cc2;
                    if (GridType == 'U' || GridType == 'E')
                        cc2 = gg.CellCenter_PB(b2, cellindex2);
                    else // AMR
                        cc2 = gg.CellCenter(b2, cellindex2);
                    if (Debug) cout<<"cc2 = "<<cc2[X]<<" "<<cc2[Y]<<" "<<cc2[Z]<<endl;

                    /// distance and length increments (mind PBCs)
                    double di=cc2[X]-cc1[X]; 
                    double dj=cc2[Y]-cc1[Y]; 
                    double dk=cc2[Z]-cc1[Z]; 
                    double distance = sqrt(di*di+dj*dj+dk*dk);

                    /// skip self-comparison and cells > MaxLength
                    if (distance <= 0 || distance > MaxLength) continue;

                    /// ========= FILL SF containers =============
                    for (int t = 0; t < NumTypes; t++)
                    {
                        switch(t)
                        {
                            case 0: /// v
                            {
                                double cs1 = 1.0, cs2 = 1.0;
                                if (temp1 && temp2) {
                                    cs1 = sqrt((double)temp1[cellindex1]*k_b/mu/m_p);
                                    cs2 = sqrt((double)temp2[cellindex1]*k_b/mu/m_p);
                                }
                                DX[t] = (double)velx2[cellindex2]/cs2 - (double)velx1[cellindex1]/cs1;
                                DY[t] = (double)vely2[cellindex2]/cs2 - (double)vely1[cellindex1]/cs1;
                                DZ[t] = (double)velz2[cellindex2]/cs2 - (double)velz1[cellindex1]/cs1;
                                break;
                            }
                            case 1: /// sqrt(rho)*v
                            {
                                double sqrtrho1 = sqrt((double)dens1[cellindex1]);
                                double sqrtrho2 = sqrt((double)dens2[cellindex2]);
                                DX[t] = sqrtrho2*(double)velx2[cellindex2] - sqrtrho1*(double)velx1[cellindex1];
                                DY[t] = sqrtrho2*(double)vely2[cellindex2] - sqrtrho1*(double)vely1[cellindex1];
                                DZ[t] = sqrtrho2*(double)velz2[cellindex2] - sqrtrho1*(double)velz1[cellindex1];
                                break;
                            }
                            case 2: /// rho^(1/3)*v
                            {
                                double pow3rho1 = pow((double)dens1[cellindex1],onethird);
                                double pow3rho2 = pow((double)dens2[cellindex2],onethird);
                                DX[t] = pow3rho2*(double)velx2[cellindex2] - pow3rho1*(double)velx1[cellindex1];
                                DY[t] = pow3rho2*(double)vely2[cellindex2] - pow3rho1*(double)vely1[cellindex1];
                                DZ[t] = pow3rho2*(double)velz2[cellindex2] - pow3rho1*(double)velz1[cellindex1];
                                break;
                            }
                            case 3: /// rho*v
                            {
                                DX[t] = (double)dens2[cellindex2]*(double)velx2[cellindex2] - (double)dens1[cellindex1]*(double)velx1[cellindex1];
                                DY[t] = (double)dens2[cellindex2]*(double)vely2[cellindex2] - (double)dens1[cellindex1]*(double)vely1[cellindex1];
                                DZ[t] = (double)dens2[cellindex2]*(double)velz2[cellindex2] - (double)dens1[cellindex1]*(double)velz1[cellindex1];
                                break;
                            }
                            case 4: /// rho (using the same containers here, so the transverse part is supposed to vanish, see below)
                            {
                                DX[t] = (double)dens2[cellindex2] - (double)dens1[cellindex1];
                                DY[t] = 0.0;
                                DZ[t] = 0.0;
                                break;
                            }
                            case 5: /// sqrt(rho)='rhotohalf' (using the same containers here, so the transverse part is supposed to vanish, see below)
                            {
                                DX[t] = sqrt((double)dens2[cellindex2]) - sqrt((double)dens1[cellindex1]);
                                DY[t] = 0.0;
                                DZ[t] = 0.0;
                                break;
                            }
                            case 6: /// ln(rho) (using the same containers here, so the transverse part is supposed to vanish, see below)
                            {
                                DX[t] = log((double)dens2[cellindex2]) - log((double)dens1[cellindex1]);
                                DY[t] = 0.0;
                                DZ[t] = 0.0;
                                break;
                            }
                            case 7: /// gb11 Flux; Exact equation (11) for F(r):
                                    /// -2*eps = S(r) + nabla_r(F(r)) in Galtier & Banerjee (2011)
                            {
                                double rho       = dens1[cellindex1];
                                double rho_prim  = dens2[cellindex2];
                                double ux        = velx1[cellindex1];
                                double uy        = vely1[cellindex1];
                                double uz        = velz1[cellindex1];
                                double ux_prim   = velx2[cellindex2];
                                double uy_prim   = vely2[cellindex2];
                                double uz_prim   = velz2[cellindex2];
                                double e         = log(rho); // assumes that cs=1 and <rho>=1 !
                                double e_prim    = log(rho_prim);
                                double d_ux      = ux_prim - ux;
                                double d_uy      = uy_prim - uy;
                                double d_uz      = uz_prim - uz;
                                double d_rhoux   = rho_prim*ux_prim - rho*ux;
                                double d_rhouy   = rho_prim*uy_prim - rho*uy;
                                double d_rhouz   = rho_prim*uz_prim - rho*uz;
                                double d_rho     = rho_prim - rho;
                                double d_e       = e_prim - e;
                                double d_rho_bar = 0.5 * (rho + rho_prim);
                                double d_e_bar   = 0.5 * (e + e_prim);
                                double square_brackets = 0.5*(d_rhoux*d_ux+d_rhouy*d_uy+d_rhouz*d_uz) + d_rho*d_e - d_rho_bar;
                                DX[t] = square_brackets * d_ux + d_e_bar * d_rhoux;
                                DY[t] = square_brackets * d_uy + d_e_bar * d_rhouy;
                                DZ[t] = square_brackets * d_uz + d_e_bar * d_rhouz;
                                break;
                            }
                            case 8: /// gb11 S(r); Exact equation (11) for S(r):
                                    /// -2*eps = S(r) + nabla_r(F(r)) in Galtier & Banerjee (2011)
                            {
                                double rho       = dens1[cellindex1];
                                double rho_prim  = dens2[cellindex2];
                                double ux        = velx1[cellindex1];
                                double uy        = vely1[cellindex1];
                                double uz        = velz1[cellindex1];
                                double ux_prim   = velx2[cellindex2];
                                double uy_prim   = vely2[cellindex2];
                                double uz_prim   = velz2[cellindex2];
                                double divu      = divv1[cellindex1];
                                double divu_prim = divv2[cellindex2];
                                double e         = log(rho); // assumes that cs=1 and <rho>=1 !
                                double e_prim    = log(rho_prim);
                                double E         = rho      * ( 0.5*(ux*ux+uy*uy+uz*uz) + e );
                                double E_prim    = rho_prim * ( 0.5*(ux_prim*ux_prim+uy_prim*uy_prim+uz_prim*uz_prim) + e_prim );
                                double uu_prim   = ux*ux_prim+uy*uy_prim+uz*uz_prim;
                                double R         = rho      * ( 0.5*uu_prim + e_prim );
                                double R_tilde   = rho_prim * ( 0.5*uu_prim + e );
                                DX[t] = divu_prim * (R - E) + divu * (R_tilde - E_prim);
                                DY[t] = 0.0;
                                DZ[t] = 0.0;
                                break;
                            }
                            default:
                            {
                                if (MyPE==0) cout << "ComputeStructureFunctions:  something is wrong with the structure function type! Exiting." << endl;
                                MPI_Finalize();
                                break;
                            }
                        }
                        /// decomposition into transverse and longitudinal parts
                        if (t <= 4) {
                          incr_long[t] = ( di*DX[t] + dj*DY[t] + dk*DZ[t] ) / distance; /// scalar product
                          incr_trsv[t] = sqrt( ( DX[t]*DX[t] + DY[t]*DY[t] + DZ[t]*DZ[t] ) - incr_long[t]*incr_long[t] ); /// Pythagoras
                        }
                        /// no decomposition
                        else {
                          incr_long[t] = DX[t];
                          incr_trsv[t] = 0.0;
                        }
                        if (t != 4) incr_long[t] = abs(incr_long[t]); // GB11 flux can be negative, but all else is abs()
                        incr_long_pow[t] = incr_long[t];
                        incr_trsv_pow[t] = incr_trsv[t];

                    } // end: types

                    /// add to the appropriate bin (nested intervals)
                    int bin1 = 0; int bin2 = NumberOfBins; int bin = 0;
                    while ((bin2 - bin1) > 1)
                    {
                        bin = bin1 + (bin2 - bin1)/2;
                        if (distance < grid[bin])
                            bin2 = bin;
                        else
                            bin1 = bin;
                    }
                    bin = bin1;

                    /// compute higher order structure functions (be cautious with numerics)
                    for (int i = 0; i < MaxStructureFunctionOrder; i++)
                      for (int t = 0; t < NumTypes; t++)
                      {
                        incr_long_pow[t] = pow(incr_long[t],(double)(i+1));
                        incr_trsv_pow[t] = pow(incr_trsv[t],(double)(i+1));
                        int BufIndex = bin*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
                        if (TotVol*incr_long_pow[t]/buf2_binsum_long[BufIndex] > numeric_epsilon)
                        {
                            buf2_bin_counter_long[BufIndex] += TotVol;
                            buf2_binsum_long     [BufIndex] += TotVol*incr_long_pow[t];
                        }
                        else // immediately reduce to buf1 and clear buf2
                        {
                            buf1_bin_counter_long          [BufIndex] += buf2_bin_counter_long[BufIndex];
                            buf1_binsum_long               [BufIndex] += buf2_binsum_long     [BufIndex];
                            buf1_numeric_error_counter_long[BufIndex]++;
                            buf2_bin_counter_long          [BufIndex] = TotVol;
                            buf2_binsum_long               [BufIndex] = TotVol*incr_long_pow[t];
                        }
                        if (TotVol*incr_trsv_pow[t]/buf2_binsum_trsv[BufIndex] > numeric_epsilon)
                        {
                            buf2_bin_counter_trsv[BufIndex] += TotVol;
                            buf2_binsum_trsv     [BufIndex] += TotVol*incr_trsv_pow[t];
                        }
                        else // immediately reduce to buf1 and clear buf2
                        {
                            buf1_bin_counter_trsv          [BufIndex] += buf2_bin_counter_trsv[BufIndex];
                            buf1_binsum_trsv               [BufIndex] += buf2_binsum_trsv     [BufIndex];
                            buf1_numeric_error_counter_trsv[BufIndex]++;
                            buf2_bin_counter_trsv          [BufIndex] = TotVol;
                            buf2_binsum_trsv               [BufIndex] = TotVol*incr_trsv_pow[t];
                        }
                      }
                    /// ==========================================

                } // end: loop over cells in 2

            } // end: loop over cells in 1

            if (velx2) delete [] velx2; if (vely2) delete [] vely2; if (velz2) delete [] velz2;
            if (dens2) delete [] dens2; if (temp2) delete [] temp2; if (divv2) delete [] divv2;

        } // end: loop over all blocks

        if (velx1) delete [] velx1; if (vely1) delete [] vely1; if (velz1) delete [] velz1;
        if (dens1) delete [] dens1; if (temp1) delete [] temp1; if (divv1) delete [] divv1;

    } // end: loop over my blocks


    /// reduce the rest of buf2 to buf1
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        buf1_bin_counter_long[BufIndex] += buf2_bin_counter_long[BufIndex];
        buf1_binsum_long     [BufIndex] += buf2_binsum_long     [BufIndex];
        buf1_bin_counter_trsv[BufIndex] += buf2_bin_counter_trsv[BufIndex];
        buf1_binsum_trsv     [BufIndex] += buf2_binsum_trsv     [BufIndex];
        // set all empty bins to 0
        if (buf1_bin_counter_long[BufIndex] <= numeric_epsilon) buf1_bin_counter_long[BufIndex] = 0;
        if (buf1_binsum_long     [BufIndex] <= numeric_epsilon) buf1_binsum_long     [BufIndex] = 0;
        if (buf1_bin_counter_trsv[BufIndex] <= numeric_epsilon) buf1_bin_counter_trsv[BufIndex] = 0;
        if (buf1_binsum_trsv     [BufIndex] <= numeric_epsilon) buf1_binsum_trsv     [BufIndex] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (MyPE==0) cout << "ComputeStructureFunctions: done. Now reducing and output..." << endl;

    /// init MPI reduction buffers
    double    *bin_counter_long           = new double    [NumBufferIndices];
    double    *bin_counter_trsv           = new double    [NumBufferIndices];
    double    *struc_funct_binsum_long    = new double    [NumBufferIndices];
    double    *struc_funct_binsum_trsv    = new double    [NumBufferIndices];
    long      *numeric_error_counter_long = new long      [NumBufferIndices];
    long      *numeric_error_counter_trsv = new long      [NumBufferIndices];
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        bin_counter_long          [BufIndex] = 0;
        bin_counter_trsv          [BufIndex] = 0;
        struc_funct_binsum_long   [BufIndex] = 0;
        struc_funct_binsum_trsv   [BufIndex] = 0;
        numeric_error_counter_long[BufIndex] = 0;
        numeric_error_counter_trsv[BufIndex] = 0;
    }

    /// Sum up CPU contributions
    MPI_Allreduce(buf1_bin_counter_long, bin_counter_long, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_bin_counter_trsv, bin_counter_trsv, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_binsum_long, struc_funct_binsum_long, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_binsum_trsv, struc_funct_binsum_trsv, NumBufferIndices, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_numeric_error_counter_long, numeric_error_counter_long, NumBufferIndices, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(buf1_numeric_error_counter_trsv, numeric_error_counter_trsv, NumBufferIndices, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

    /// clean-up
    delete[] buf1_bin_counter_long; buf1_bin_counter_long = 0;
    delete[] buf1_bin_counter_trsv; buf1_bin_counter_trsv = 0;
    delete[] buf1_binsum_long; buf1_binsum_long = 0;
    delete[] buf1_binsum_trsv; buf1_binsum_trsv = 0;
    delete[] buf1_numeric_error_counter_long; buf1_numeric_error_counter_long = 0;
    delete[] buf1_numeric_error_counter_trsv; buf1_numeric_error_counter_trsv = 0;
    delete[] buf2_bin_counter_long; buf2_bin_counter_long = 0;
    delete[] buf2_bin_counter_trsv; buf2_bin_counter_trsv = 0;
    delete[] buf2_binsum_long; buf2_binsum_long = 0;
    delete[] buf2_binsum_trsv; buf2_binsum_trsv = 0;


    /// divide binsum by bin_counter
    for (int BufIndex = 0; BufIndex < NumBufferIndices; BufIndex++)
    {
        if (bin_counter_long[BufIndex] > 0)
            struc_funct_binsum_long[BufIndex] /= bin_counter_long[BufIndex];
        else
            struc_funct_binsum_long[BufIndex] = 0.0;
        if (bin_counter_trsv[BufIndex] > 0)
            struc_funct_binsum_trsv[BufIndex] /= bin_counter_trsv[BufIndex];
        else
            struc_funct_binsum_trsv[BufIndex] = 0.0;
    }

    /// print out useful information and output
    if (MyPE==0)
    {
        double ReSumCounter_long[NumTypes][MaxStructureFunctionOrder];
        double ReSumCounter_trsv[NumTypes][MaxStructureFunctionOrder];
        for (int i = 0; i < MaxStructureFunctionOrder; i++)
          for (int t = 0; t < NumTypes; t++)
          {
            ReSumCounter_long[t][i] = 0;
            ReSumCounter_trsv[t][i] = 0;
          }
        for (int t = 0; t < NumTypes; t++)
        {
         switch(t)
         {
          case 0: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for pure VELOCITY structure functions.          ********************" << endl; break; }
          case 1: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for SQRT(RHO)*VELOCITY structure functions.     ********************" << endl; break; }
          case 2: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for RHO^(1/3)*VELOCITY structure functions.     ********************" << endl; break; }
          case 3: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for RHO*VELOCITY structure functions.           ********************" << endl; break; }
          case 4: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for RHO structure functions.                    ********************" << endl; break; }
          case 5: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for RHOTOHALF structure functions.              ********************" << endl; break; }
          case 6: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for LN(RHO) structure functions.                ********************" << endl; break; }
          case 7: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for GB11 FLUX structure functions.              ********************" << endl; break; }
          case 8: { cout << endl << "ComputeStructureFunctions:  *********************** statistics and summary for GB11 S(r) structure functions.              ********************" << endl; break; }
         }
         for (int b = 0; b < NumberOfBins; b++)
         {
          cout << "bin_counter_long   [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << bin_counter_long[BufIndex];
            ReSumCounter_long[t][i] += bin_counter_long[BufIndex];
          }
          cout << endl;
          cout << "error_counter_long [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << numeric_error_counter_long[BufIndex];
          }
          cout << endl;
          cout << "bin_counter_trsv   [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << bin_counter_trsv[BufIndex];
            ReSumCounter_trsv[t][i] += bin_counter_trsv[BufIndex];
          }
          cout << endl;
          cout << "error_counter_trsv [" << setw(3) << b << "]=";
          for (int i = 0; i < MaxStructureFunctionOrder; i++)
          {
            int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
            cout << setw(14) << numeric_error_counter_trsv[BufIndex];
          }
          cout << endl;
         }
         for (int i = 0; i < MaxStructureFunctionOrder; i++)
         {
            cout << "Resummed total number of samples for longitudinal structure functions [order="<<setw(2)<<i+1<<"] = " << ReSumCounter_long[t][i] << endl;
            cout << "Resummed total number of samples for transverse   structure functions [order="<<setw(2)<<i+1<<"] = " << ReSumCounter_trsv[t][i] << endl;
         }
        }

        /// prepare OutputFileHeader
        OutputFileHeader.resize(0);
        stringstream dummystream;
        char colichar[4]; char sfochar[4];
        string colstr = ""; string sfostr = "";
        dummystream << setw(30) << left << "#00_BinIndex";
        dummystream << setw(30) << left << "#01_GridStag";
        dummystream << setw(30) << left << "#02_Grid";
        for (int i = 0; i < MaxStructureFunctionOrder; i++)
        {
          sprintf(sfochar, "%2.2d", i+1); sfostr = sfochar;
          sprintf(colichar, "%2.2d", 3+4*i+0); colstr = colichar;
          dummystream << setw(30) << left << "#"+colstr+"_NP(long,order="+sfostr+")";
          sprintf(colichar, "%2.2d", 3+4*i+1); colstr = colichar;
          dummystream << setw(30) << left << "#"+colstr+"_SF(long,order="+sfostr+")";
          sprintf(colichar, "%2.2d", 3+4*i+2); colstr = colichar;
          dummystream << setw(30) << left << "#"+colstr+"_NP(trsv,order="+sfostr+")";
          sprintf(colichar, "%2.2d", 3+4*i+3); colstr = colichar;
          dummystream << setw(30) << left << "#"+colstr+"_SF(trsv,order="+sfostr+")";
        }
        OutputFileHeader.push_back(dummystream.str());
        dummystream.clear();
        dummystream.str("");
        /// OUTPUT in files (needs to be here because of the different types)
        string OutputFilename = "";
        for (int t = 0; t < NumTypes; t++)
        {
            /// resize and fill WriteOutTable
            WriteOutTable.resize(NumberOfBins); /// structure function output has NumberOfBins lines
            for (unsigned int i = 0; i < WriteOutTable.size(); i++)
                WriteOutTable[i].resize(3+4*MaxStructureFunctionOrder); /// structure function output has ... columns
            for (int b = 0; b < NumberOfBins; b++)
            {
                WriteOutTable[b][0] = b;
                WriteOutTable[b][1] = grid_stag[b];
                WriteOutTable[b][2] = grid     [b];
                for (int i = 0; i < MaxStructureFunctionOrder; i++)
                {
                    int BufIndex = b*NumTypes*MaxStructureFunctionOrder+i*NumTypes+t;
                    WriteOutTable[b][3+4*i+0] = bin_counter_long       [BufIndex];
                    WriteOutTable[b][3+4*i+1] = struc_funct_binsum_long[BufIndex];
                    WriteOutTable[b][3+4*i+2] = bin_counter_trsv       [BufIndex];
                    WriteOutTable[b][3+4*i+3] = struc_funct_binsum_trsv[BufIndex];
                }
            }
            switch (t)
            {
                case 0: { OutputFilename = OutputPath + "/" + inputfile + "_sf_vels.dat"; break; }
                case 1: { OutputFilename = OutputPath + "/" + inputfile + "_sf_sqrtrho.dat"; break; }
                case 2: { OutputFilename = OutputPath + "/" + inputfile + "_sf_rho3.dat"; break; }
                case 3: { OutputFilename = OutputPath + "/" + inputfile + "_sf_rhov.dat"; break; }
                case 4: { OutputFilename = OutputPath + "/" + inputfile + "_sf_rho.dat"; break; }
                case 5: { OutputFilename = OutputPath + "/" + inputfile + "_sf_rhotohalf.dat"; break; }
                case 6: { OutputFilename = OutputPath + "/" + inputfile + "_sf_lnrho.dat"; break; }
                case 7: { OutputFilename = OutputPath + "/" + inputfile + "_sf_gb11flux.dat"; break; }
                case 8: { OutputFilename = OutputPath + "/" + inputfile + "_sf_gb11s.dat"; break; }
            }
            WriteOutAnalysedData(OutputFilename);

        } // end: loop over types

    } // end: MyPE==0

    /// clean-up
    delete[] bin_counter_long; bin_counter_long = 0;
    delete[] bin_counter_trsv; bin_counter_trsv = 0;
    delete[] struc_funct_binsum_long; struc_funct_binsum_long = 0;
    delete[] struc_funct_binsum_trsv; struc_funct_binsum_trsv = 0;
    delete[] numeric_error_counter_long; numeric_error_counter_long = 0;
    delete[] numeric_error_counter_trsv; numeric_error_counter_trsv = 0;

    if (MyPE==0) cout << "ComputeStructureFunctions: exiting." << endl;
}


/** -------------------- WriteOutAnalysedData ---------------------------------
 **  Writes out a variable table of data and a FileHeader to a specified file
 ** --------------------------------------------------------------------------- */
void WriteOutAnalysedData(const string OutputFilename)
{
    /// open output file
    ofstream Outputfile(OutputFilename.c_str());

    /// check for file
    if (!Outputfile)
    {
        cout << "WriteOutAnalysedData:  File system error. Could not create '" << OutputFilename.c_str() << "'."<< endl;
        MPI_Finalize();
    }
    /// write data to output file
    else
    {
        cout << "WriteOutAnalysedData:  Writing output file '" << OutputFilename.c_str() << "' ..." << endl;

        for (unsigned int row = 0; row < OutputFileHeader.size(); row++)
        {
            Outputfile << setw(61) << left << OutputFileHeader[row] << endl;      /// header
            if (Debug) cout << setw(61) << left << OutputFileHeader[row] << endl;
        }
        for (unsigned int row = 0; row < WriteOutTable.size(); row++)                  /// data
        {
            for (unsigned int col = 0; col < WriteOutTable[row].size(); col++)
            {
                Outputfile << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
                if (Debug) cout << scientific << setw(30) << left << setprecision(8) << WriteOutTable[row][col];
            }
            Outputfile << endl; if (Debug) cout << endl;
        }

        Outputfile.close();
        Outputfile.clear();

        cout << "WriteOutAnalysedData:  done!" << endl;
    }
} /// =======================================================================


/** ------------------------- ParseInputs ----------------------------
 **  Parses the command line Arguments
 ** ------------------------------------------------------------------ */
int ParseInputs(const vector<string> Argument)
{
    stringstream dummystream;

    /// read tool specific options
    if (Argument.size() < 2)
    {
        if (MyPE==0) { cout << endl << "ParseInputs: Invalid number of arguments." << endl; }
        return -1;
    }
    inputfile = Argument[1];

    for (unsigned int i = 2; i < Argument.size(); i++)
    {
        if (Argument[i] != "" && Argument[i] == "-n")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> n_samples; dummystream.clear();
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-ncells_pb")
        {
            if (Argument.size()>i+1) {
                dummystream << Argument[i+1]; dummystream >> ncells_pseudo_blocks; dummystream.clear();
                ncells_pseudo_blocks_set = true;
            } else return -1;
        }
        if (Argument[i] != "" && Argument[i] == "-opath")
        {
            if (Argument.size()>i+1) OutputPath = Argument[i+1]; else return -1;
        }

    } // loop over all args

    /// print out parsed values
    if (MyPE==0) {
        cout << "ParseInputs: Command line arguments: ";
        for (unsigned int i = 0; i < Argument.size(); i++) cout << Argument[i] << " ";
        cout << endl;
    }
    return 0;

} // end: ParseInputs()


/** --------------------------- HelpMe -------------------------------
 **  Prints out helpful usage information to the user
 ** ------------------------------------------------------------------ */
void HelpMe(void)
{
    if (MyPE==0) {
        cout << endl
        << "Syntax:" << endl
        << " strufu_mpi <filename> [<OPTIONS>]" << endl << endl
        << "   <OPTIONS>:           " << endl
        << "     -n <num_samples>        : total number of sampling pairs" << endl
        << "     -ncells_pb <num_cells>  : number of cells in pseudo blocks (default: as in file)" << endl
        << "     -opath <path>           : specify output path" << endl
        << endl
        << "Example: strufu_mpi DF_hdf5_plt_cnt_0020 -n 1e6"
        << endl << endl;
    }
}
