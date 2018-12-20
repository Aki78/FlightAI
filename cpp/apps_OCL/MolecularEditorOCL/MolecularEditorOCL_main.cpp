
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>


#include "testUtils.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include "Draw.h"
#include "Draw3D.h"
#include "SDL_utils.h"
#include "Solids.h"

#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"
#include "quaternion.h"

#include "raytrace.h"
#include "Molecule.h"
#include "MMFF.h"
#include "MMFFBuilder.h"

#include "IO_utils.h"


#include "geom3D.h"

//#include "RBMMFF.h"
#include "DynamicOpt.h"

#include "AtomicConfiguration.h"
#include "DistanceHierarchy.h"

#include "AppSDL2OGL_3D.h"
#include "MolecularDraw.h"

#include "MolecularWorl2OCL.h"

/*

TO DO:
 - save geom to .xyz
 - save geom to .pdb or .mol
 - add charges (read from .pdb)
 - torsion angles
 - add better model of substrate
 - include rigid body molecules (from MoleculerWorld )
 - Brute force non-bonded interactions by OpenCL
 - add some editation capabilities

 TODO Corrections:
 - vdW distances seems to be too close
 - some bonds too long
 - correct angular forcefield to repdesent kinked groups ( e.g. -OH )
*/

std::vector<Vec3d> iso_points;
int isoOgl;

Vec3d PPpos0 = (Vec3d){1.3,1.7, 1.5};

Vec3d testREQ,testPLQ;

// ==========================
// AppMolecularEditorOCL
// ==========================75

void drawAtomsF8( int n, float8 * atoms, float sc, int oglSphere ){
    for(int i=0; i<n; i++){
        float* atomi = ((float*)(atoms+i));
        float r = atomi[4]*sc;
        float q = atomi[6];
        glColor3f( 0.5+q, 0.5, 0.5-q );
        Draw3D::drawShape( *(Vec3f*)atomi, {0.0,0.0,0.0,1.0}, (Vec3f){r,r,r}, oglSphere );
    }
}

void drawAtomsForces( int n, float8 * atoms, Vec3f * fatoms, float rsc, float fsc ){
    Quat4f* ps = (Quat4f*) atoms;
    for(int i=0; i<n; i++){
        const Vec3f& p = ps[i*2].f;
        printf( "atoms[%i] f(%g,%g,%g)  \n", i, fatoms[i].x, fatoms[i].y, fatoms[i].z );
        Draw3D::drawPointCross( p, rsc           );
        Draw3D::drawVecInPos  ( fatoms[i]*fsc, p );
    }
}

void drawRigidMolAtomForce( const Vec3f& pos, const Quat4f& qrot, const Vec3f& fpos, const Vec3f& torq, int n, const float8 * atom0s,  float rsc, float fsc ){
    Mat3f mrot; qrot.toMatrix(mrot);
    for(int i=0; i<n; i++){
    
        Vec3f Mp;
        //p = *((Vec3f*)(atom0s+j));
        mrot.dot_to_T( *((Vec3f*)(atom0s+i)), Mp );
        Mp.add( pos );
        
        Vec3f f; 
        f.set_cross(torq,Mp);
        f.add(fpos);
       
        //Draw3D::drawShape( pi, {0.0,0.0,0.0,1.0}, (Vec3f){r,r,r}, oglSphere );
        Draw3D::drawPointCross( Mp, rsc   );
        Draw3D::drawVecInPos  ( f*fsc, Mp );
    }
}

void drawRigidMolAtomCOG( const Vec3f& pos, const Quat4f& qrot, int n, const float8 * atom0s,  float rsc ){
    Mat3f mrot; qrot.toMatrix(mrot);
    for(int i=0; i<n; i++){
        Vec3f Mp;
        //p = *((Vec3f*)(atom0s+j));
        mrot.dot_to( *((Vec3f*)(atom0s+i)), Mp );
        Mp.add( pos );       
        //Draw3D::drawShape( pi, {0.0,0.0,0.0,1.0}, (Vec3f){r,r,r}, oglSphere );
        Draw3D::drawPointCross( Mp, rsc   );
        Draw3D::drawLine      ( Mp, pos   );
    }
}

void drawRigidMolSystem(const RigidMolecularWorldOCL& clworld, int isystem ){
    int isoff   = isystem * clworld.nMols;
    int atom_count = 0;
    const float8* atom0s   = clworld.atomsInTypes.data();
    Quat4f* posi     = (Quat4f*)(clworld. poses+isoff);
    //Quat4f* fsi      = (Quat4f*)(clworld.fposes+isoff);
    for(int imol=0; imol<clworld.nMols; imol++){
        //float* posi     = (float*)(clworld. poses+isoff+imol);
        //float* fsi      = (float*)(clworld.fposes+isoff+imol);
        const int2& m2a = clworld.mol2atoms[imol];
        //printf( "isystem %i imol %i m2a (%i,%i) atom_count %i %i \n", isystem, imol, m2a.x, m2a.y, atom_count, atom_count );
        //frag2atoms( *((Vec3f*)(posi)), *((Quat4f*)(posi+4)), m2a.y, atom0s+m2a.x, atoms+atom_count );
        //drawRigidMolAtomForce( posi[0].f, posi[1], fsi[0].f, fsi[1].f, m2a.y, atom0s+m2a.x,  0.25, 10.0 );
        drawRigidMolAtomCOG( posi[0].f, posi[1], m2a.y, atom0s+m2a.x,  0.25 );
                
        posi+=2;
        //fsi +=2;
        //atom_count += m2a.y;
    }
    //return atom_count;
}

void drawRigidMolSystemForceTorq(const RigidMolecularWorldOCL& clworld, int isystem, float fsc, float tsc ){
    int isoff   = isystem * clworld.nMols;
    int atom_count = 0;
    const float8* atom0s   = clworld.atomsInTypes.data();
    Quat4f* posi     = (Quat4f*)(clworld. poses+isoff);
    Quat4f* fsi      = (Quat4f*)(clworld.fposes+isoff);
    for(int imol=0; imol<clworld.nMols; imol++){
        //Draw3D::drawPointCross( posi[0].f, rsc   );
        glColor3f(1.0,0.0,0.0); Draw3D::drawVecInPos( fsi[0].f*fsc, posi[0].f );  // force
        glColor3f(0.0,0.0,1.0); Draw3D::drawVecInPos( fsi[1].f*tsc, posi[0].f );  // torq    
        posi+=2;
        fsi +=2;
    }
}

class AppMolecularEditorOCL : public AppSDL2OGL_3D { public:
	//Molecule    mol;
	MMFFparams  params;
    MMFF        world;
    MMFFBuilder builder;
    
    OCLsystem* cl;
    GridFF_OCL              gridFFocl;
    RigidMolecularWorldOCL  clworld;

    FastAtomicMetric atomdist;
    AtomicConfiguration conf1;
    DistanceHierarchy<AtomicConfiguration> database;

    DynamicOpt  opt;

    int     fontTex;
    int     ogl_sph;

    char str[256];

    AtomicManipulator manipulator;

    Vec3d ray0;
    int ipicked  = -1, ibpicked = -1;
    int perFrame =  50;

    Vec3d cursor3D=(Vec3d){0.0,0.0,0.0};

    double drndv =  10.0;
    double drndp =  0.5;

    double  atomSize = 0.25;

    int itest=0;
    
    
    // TEMP
    int isystem = 0;
    float8* atoms_tmp=0;  // = new float8[100];
    Vec3f* fatoms_tmp=0;
    int atom_count=0;     //= clworld.system2atoms( 0, atoms );


    Quat4f qrot = Quat4fBack;


    // ==== Functions

    void genNewManipul(int i);
    bool manipulation();
    
    void stepCPU( double& F2, bool randomConf = false );
    void drawCPU();
    

	virtual void draw   ()  override;
	virtual void drawHUD()  override;
	//virtual void mouseHandling( )  = override;
	virtual void eventHandling   ( const SDL_Event& event  ) override;
	virtual void keyStateHandling( const Uint8 *keys ) override;

	AppMolecularEditorOCL( int& id, int WIDTH_, int HEIGHT_ );

    void initRigidSubstrate();
};

void AppMolecularEditorOCL::initRigidSubstrate(){

    // ---- Rigid Substrate
    //world.substrate.init( (Vec3i){100,100,100}, (Mat3d){ 10.0,0.0f,0.0f,  0.0,10.0f,0.0f,  0.0,0.0f,10.0f }, (Vec3d){-5.0,-5.0,-5.0} );
    
    printf( "params.atypNames:\n" );
    for(auto kv : params.atypNames) { printf(" %s %i \n", kv.first.c_str(), kv.second ); }
    //exit(0);
    //world.substrate.grid.n    = (Vec3i){120,120,200};
    world.gridFF.grid.n    = (Vec3i){60,60,100};
    //world.substrate.grid.n    = (Vec3i){12,12,20};
    world.gridFF.grid.pos0 = (Vec3d){0.0d,0.0d,0.0d};
    world.gridFF.loadCell ( "inputs/cel.lvs" );
    //world.gridFF.loadCell ( "inputs/cel_2.lvs" );
    world.gridFF.grid.printCell();
    //world.gridFF.loadXYZ  ( "inputs/answer_Na_L1.xyz", params );
    //world.gridFF.loadXYZ  ( "inputs/NaCl_sym.xyz", params );
    world.gridFF.loadXYZ  ( "inputs/NaCl_wo4.xyz", params );
    //world.gridFF.loadXYZ  ( "inputs/NaCl_sym_Na_add.xyz", params );
    //world.gridFF.loadXYZ  ( "inputs/NaCl_sym_Cl_vac.xyz", params );
    //world.gridFF.loadXYZ  ( "inputs/NaCl_sym_Na_vac.xyz", params );
    //world.gridFF.loadXYZ  ( "inputs/Xe_instead_Na.xyz", params );
    //world.gridFF.loadXYZ( "inputs/Cl.xyz", params );
    world.translate( {0.0,0.0,4.5} );


    world.genPLQ();
    world.gridFF.allocateFFs();
    //world.gridFF.evalGridFFs( {0,0,0} );

    //world.gridFF.setAtoms( int natoms, Vec3d * apos_, Vec3d * REQs_ );

    gridFFocl.evalGridFFs(world.gridFF, {1,1,1} ); DEBUG    
    
    /*
    world.gridFF.grid.saveXSF( "FFPauli_x.xsf",  world.gridFF.FFPauli, 0 );
    world.gridFF.grid.saveXSF( "FFPauli_y.xsf",  world.gridFF.FFPauli, 1 );
    world.gridFF.grid.saveXSF( "FFPauli_z.xsf",  world.gridFF.FFPauli, 2 );
    //world.gridFF.grid.saveXSF( "FFPauli_e.xsf",  world.gridFF.FFPauli, 3 );
    
    world.gridFF.grid.saveXSF( "FFLondon_x.xsf", world.gridFF.FFLondon, 0 );
    world.gridFF.grid.saveXSF( "FFLondon_y.xsf", world.gridFF.FFLondon, 1 );
    world.gridFF.grid.saveXSF( "FFLondon_z.xsf", world.gridFF.FFLondon, 2 );
    //world.gridFF.grid.saveXSF( "FFLondon_e.xsf", world.gridFF.FFLondon, 3 );
    
    world.gridFF.grid.saveXSF( "FFelec_x.xsf",   world.gridFF.FFelec, 0 );
    world.gridFF.grid.saveXSF( "FFelec_y.xsf",   world.gridFF.FFelec, 1 );
    world.gridFF.grid.saveXSF( "FFelec_z.xsf",   world.gridFF.FFelec, 2 );
    //world.gridFF.grid.saveXSF( "FFelec_e.xsf",   world.gridFF.FFelec, 3 );
    */
    
    int iatom = 11;
    //testREQ = (Vec3d){ 2.181, 0.0243442, 0.0}; // Xe
    //testREQ = (Vec3d){ 1.487, 0.0006808, 0.0}; // H
    testREQ = (Vec3d){ 1.487, sqrt(0.0006808), 0.0 };
    testPLQ = REQ2PLQ( testREQ, world.gridFF.alpha );//
    printf( "testREQ   (%g,%g,%g) -> PLQ (%g,%g,%g) \n",        testREQ.x, testREQ.y, testREQ.z, testPLQ.x, testPLQ.y, testPLQ.z   );
    printf( "aREQs[%i] (%g,%g,%g) -> PLQ (%g,%g,%g) \n", iatom, world.aREQ[iatom].x, world.aREQ[iatom].y, world.aREQ[iatom].z, world.aPLQ[iatom].x, world.aPLQ[iatom].y, world.aPLQ[iatom].z );
    Vec3d * FFtot = new Vec3d[world.gridFF.grid.getNtot()];
    world.gridFF.evalCombindGridFF( testREQ, FFtot );
    
    /*
    saveXSF( "FFtot_x.xsf", world.gridFF.grid, FFtot, 0, world.gridFF.natoms, world.gridFF.apos, world.gridFF.atypes );
    saveXSF( "FFtot_y.xsf", world.gridFF.grid, FFtot, 1, world.gridFF.natoms, world.gridFF.apos, world.gridFF.atypes );
    saveXSF( "FFtot_z.xsf", world.gridFF.grid, FFtot, 2, world.gridFF.natoms, world.gridFF.apos, world.gridFF.atypes );
    */ 
     
    isoOgl = glGenLists(1);
    glNewList(isoOgl, GL_COMPILE);
        //getIsovalPoints_a( world.gridFF.grid, 0.1, FFtot, iso_points );
        //renderSubstrate( iso_points.size(), &iso_points[0], GL_POINTS );
        renderSubstrate_( world.gridFF.grid, FFtot, world.gridFF.FFelec, 0.1, true );
        //renderSubstrate_( world.gridFF.grid, world.gridFF.FFPauli, world.gridFF.FFelec, 0.01, true );
        Draw3D::drawAxis(1.0);
    glEndList();

    cam.pos.z = +5.0;

}

AppMolecularEditorOCL::AppMolecularEditorOCL( int& id, int WIDTH_, int HEIGHT_ ) : AppSDL2OGL_3D( id, WIDTH_, HEIGHT_ ) {

    cl = new OCLsystem();  DEBUG
    cl->init();
    gridFFocl.init( cl, "cl/FF.cl" ); DEBUG
    clworld  .init( cl, "cl/relaxMolecules.cl" ); DEBUG
    
    fontTex = makeTexture( "common_resources/dejvu_sans_mono_RGBA_inv.bmp" );

    ogl_sph = glGenLists(1);
    glNewList( ogl_sph, GL_COMPILE );
        Draw3D::drawSphere_oct( 3, 1.0, {0.0,0.0,0.0} );
        //Draw3D::drawSphere_oct( 3, 0.25, {0.0,0.0,0.0} );
    glEndList();

    //qCamera.set( 0.0,0.0,0.0,1.0 );  // bottom view
    //qCamera.set( 0.0,0.0,1.0,0.0 );  // bottom view
    //qCamera.set( 0.0,1.0,0.0,0.0 );  // top view  x=-x, y=y,
    qCamera.set( 1.0,0.0,0.0,0.0 );    // top view  x=x, y=-y,
    //qCamera.set( 0.70710678118,0.0,0.0,0.70710678118 ); // side down
    //qCamera.set( -0.70710678118,0.0,0.0,0.70710678118 ); // x=x, z=y,  y=-y,
    //qCamera.set( 0.0, -0.70710678118,0.0,0.70710678118 ); // z=-x, y=y
    //qCamera.set( 0.0, +0.70710678118,0.0,0.70710678118 ); // z=+x, y=y
    //qCamera.set( 0.0,0.0, +0.70710678118, 0.70710678118 ); // y=-x, x=y
    //qCamera.set( 0.0,0.0, -0.70710678118, 0.70710678118 ); // y=x, x=-y

    //AtomType atyp;
    //atyp.fromString( "CA 6 4 4 1 2.00 0.09 0x11EEAA" );
    builder.params = &params;
    params.loadAtomTypes( "common_resources/AtomTypes.dat" );
    params.loadBondTypes( "common_resources/BondTypes.dat" );
    //for(auto kv : params.atypNames) { printf( ">>%s<< %i \n", kv.first.c_str(), kv.second ); };
    char str[1024];
    printf( "type %s \n", params.atypes[ params.atypNames.find( "C" )->second ].toString( str ) );
    printf( "type %s \n", params.atypes[ params.atypNames.find( "H" )->second ].toString( str ) );
    printf( "type %s \n", params.atypes[ params.atypNames.find( "O" )->second ].toString( str ) );
    printf( "type %s \n", params.atypes[ params.atypNames.find( "N" )->second ].toString( str ) );
    DEBUG
    /*
    auto it = params.atypNames.find( "C" );
    if( it != params.atypNames.end() ){
        //printf( "type CA %i \n", it->second );
        printf( "type %i %s \n", it->second, params.atypes[ it->second ].toString( str ) );
    }else{
        printf("not found\n");
    }
    */

    //mol.atypNames = &params.atypNames;
    //exit(0);

    /*
    // ---- Rigid Body Molecules
    //mol.loadXYZ( "inputs/water_ax.xyz" );                           
    //mol.loadXYZ( "inputs/water_ax_q0.xyz" );                        
    //mol.loadXYZ( "inputs/OH_ax.xyz" );                              
    mol.loadXYZ( "inputs/water_T5_ax.xyz" );    mol.printAtomInfo();   DEBUG
    params.assignREs( mol.natoms, mol.atomType, mol.REQs );
    Mat3d rot; rot.setOne();
    builder.insertMolecule( &mol, {0.0,0.0,8.0}, rot, true );          DEBUG
    //builder.insertMolecule( &mol, {2.0,2.0,8.0}, rot, true );          DEBUG
    //builder.insertMolecule( &mol, {0.0,0.0,4.0}, rot, true );          DEBUG
    //builder.insertMolecule( &mol, {4.0,0.0,4.0}, rot, true );
    //builder.insertMolecule( &mol, {0.0,4.0,4.0}, rot, true );
    //builder.insertMolecule( &mol, {4.0,4.0,4.0}, rot, true );
    clworld.addMolType( mol );

    mol.loadXYZ( "inputs/NaIon.xyz" ); mol.printAtomInfo();
    params.assignREs( mol.natoms, mol.atomType, mol.REQs );
    //builder.insertMolecule( &mol, {4.0,6.0,5.0}, rot, false );
    //builder.insertMolecule( &mol, {4.0,4.0,2.0}, rot, false );
    //builder.insertMolecule( &mol, {4.0,8.0,2.0}, rot, false );
    clworld.addMolType( mol );

    mol.loadXYZ( "inputs/ClIon.xyz" ); mol.printAtomInfo();
    params.assignREs( mol.natoms, mol.atomType, mol.REQs );
    //builder.insertMolecule( &mol, {2.0,6.0,2.0}, rot, false );
    //builder.insertMolecule( &mol, {6.0,6.0,2.0}, rot, false );
    clworld.addMolType( mol );
    
    mol.loadXYZ( "inputs/OHion.xyz" ); mol.printAtomInfo();
    params.assignREs( mol.natoms, mol.atomType, mol.REQs );
    //builder.insertMolecule( &mol, {2.0,6.0,2.0}, rot, false );
    //builder.insertMolecule( &mol, {6.0,6.0,2.0}, rot, false );
    clworld.addMolType( mol );
    */
    
    DEBUG
    builder.loadMolType( "inputs/water_T5_ax.xyz", "H2O" );
    builder.loadMolType( "inputs/NaIon.xyz", "Na+" );
    builder.loadMolType( "inputs/ClIon.xyz", "Cl-" );
    builder.loadMolType( "inputs/OHion.xyz", "OH-" ); 
    DEBUG
    for( Molecule* mol : builder.molTypes ){
        //mol->atypNames = &params.atypNames;
        mol->printAtomInfo();
        params.assignREs( mol->natoms, mol->atomType, mol->REQs );
        clworld.addMolType( *mol );
    }
    DEBUG
    Mat3d rot; rot.setOne();
    //builder.insertMolecule( "OH-", {0.0,0.0,8.0}, rot, true );
    builder.insertMolecule( "H2O", {0.0,0.0,8.0}, rot, true );
    DEBUG

    world.printAtomInfo();
    builder.toMMFF( &world );                                 DEBUG
    world.printAtomInfo(); //exit(0);
    //world.allocFragment( nFrag );
    //opt.bindArrays( 8*world.nFrag, (double*)world.poses, new double[8*world.nFrag], (double*)world.poseFs ); 
    //opt.bindArrays( 8*world.nFrag, world.poses, world.poseVs, world.poseFs );
    world.allocateDyn(); 
    world.initDyn();     
    opt.bindArrays( world.nDyn, world.dynPos, world.dynVel, world.dynForce ); DEBUG
    opt.setInvMass( 1.0 );  
    opt.cleanVel  ( );      
    //exit(0);
    printf("POSE_pos   : \n"); printPoses( world.nFrag, world.poses  );
    printf("POSE_Force : \n"); printPoses( world.nFrag, world.poseFs );
    //exit(0);

    DEBUG

    initRigidSubstrate();
    
    DEBUG
    
    //int nMols  = 1;
    int nMols    = world.nFrag;
    int nSystems = 1;
    
    //clworld.prepareBuffers( nSystems, nMols, world.gridFF.grid.n, world.gridFF.FFPauli_f, world.gridFF.FFLondon_f, world.gridFF.FFelec_f );
    clworld.prepareBuffers( nSystems, nMols, world.gridFF );
    
    DEBUG
    
    printf( " SETUP CLWORLD nSystem %i nMols %i \n", nSystems, nMols );
    
    int i=0;
    float span = 5.0;
    for(int isys=0; isys<nSystems; isys++){
        Quat4f*  ps = (Quat4f*)clworld.poses;
        Quat4d* wps = (Quat4d*)  world.poses;
        for(int imol=0; imol<nMols; imol++){
            /*
            //clworld.mol2atoms[i].x = 0;  // molecule type 0 - watter
            //clworld.mol2atoms[i].y = 3;  // 3 atoms per molecule 
            //clworld.setMolInstance( isys, imol, 0 );
            //clworld.mol2atoms[i] = clworld.molTypes[0]; // water 3 atoms
            clworld.mol2atoms[i] = clworld.molTypes[1]; // Na 1 atom
            //clworld.mol2atoms[i] = clworld.molTypes[2]; // Cl 3 atoms
            
            *( (Vec3f* )&clworld.poses[i].x  ) = (Vec3f){ randf(-span,span), randf(-span,span), randf(0.0,1.0) } + (Vec3f){0.0, 0.0, 7.0};
            //( (Vec3f* )&clworld.poses[i].x  )->set( randf(-1.0,1.0),randf(-1.0,1.0),randf(-1.0,1.0) );
            //( (Quat4f*)&clworld.poses[i].hx )->setOne();
            ( (Quat4f*)&clworld.poses[i].hx )->setRandomRotation();
            //printf( "  \n",     );
            */
            
            //double *  poses   = NULL; // rigd body pose of molecule (pos,qRot);
            //double *  poseFs  = NULL; //
            clworld.mol2atoms[i] = clworld.molTypes[0];
            //clworld.mol2atoms[i] = clworld.molTypes[3];
            ps[0] = (Quat4f)wps[0];
            ps[1] = (Quat4f)wps[1];
            
            printf( "  world pose p(%5.5e,%5.5e,%5.5e) f(%5.5e,%5.5e,%5.5e,%5.5e) \n",  wps[0].f.x, wps[0].f.y, wps[0].f.z,  wps[1].x, wps[1].x, wps[1].x, wps[1].w );
            printf( "clworld pose p(%5.5e,%5.5e,%5.5e) f(%5.5e,%5.5e,%5.5e,%5.5e) \n",   ps[0].f.x,  ps[0].f.y,  ps[0].f.z,   ps[1].x,  ps[1].x,  ps[1].x,  ps[1].w );
            ps +=2;
            wps+=2;
            i++;
        }
    }
    
    //exit(0);
    
    DEBUG
    
    clworld.upload_mol2atoms();  DEBUG
    clworld.upload_poses();      DEBUG  // PO SEM DOBRE
    clworld.setupKernel( world.gridFF.grid, world.gridFF.alpha ); 
    clworld.task_getForceRigidSystemSurfGrid->enque();  DEBUG
    clworld.download_poses();    DEBUG
    clworld.download_fposes();   DEBUG
    clFinish(cl->commands);      DEBUG;
   
    DEBUG
    
    atoms_tmp  = new float8[1000];
    fatoms_tmp = new Vec3f [1000];
    atom_count = clworld.system2atoms( isystem, atoms_tmp );
    
    manipulator.bindAtoms(world.natoms, world.apos, world.aforce ); 
    manipulator.realloc(1);                                        
    manipulator.goalSpan.set(5.0,5.0,1.0);                          
    manipulator.genGoals();                                         

    manipulator.nenabled = 10;
    manipulator.enabled = new int[manipulator.nenabled];
    std::memcpy( manipulator.enabled, (const int[]){0,1,2,3,4,5,6,7,8,9}, manipulator.nenabled*sizeof(int) );

    DEBUG
    //exit(0);

/*
    /// =========== Configurtions
    conf1.bind( 5, world.atypes, world.apos );
    atomdist.copyOf(conf1);
    atomdist.initRuler( world.Collision_box.a+(Vec3d){-2.0,-2.0,-2.0}, world.Collision_box.b+(Vec3d){3.0,3.0,3.0}, 2.0 );
    printf( "atomdist.ruler: %i (%i,%i,%i)\n ", atomdist.ruler.ntot, atomdist.ruler.n.x, atomdist.ruler.n.y, atomdist.ruler.n.z );

    //atomdist.toCells( 0.5 );
    atomdist.toCells();
    conf1.pos[0].add(0.1,0.0,0.0);
    for(int i=0; i<5; i++){ printf( "== %i %i(%f,%f,%f) | %i %i(%f,%f,%f)\n", i, world.atypes[i], world.apos[i].x,world.apos[i].y,world.apos[i].z,
                                                                              atomdist.types[i],  atomdist.pos[i].x, atomdist.pos[i].y, atomdist.pos[i].z  ); }
    double rTrue =((AtomicConfiguration)atomdist).dist(conf1);
    double rFast = atomdist.dist( conf1 );

    printf( "dist = %f %f \n", rTrue, rFast );
 */

    printf( "SETUP DONE !\n" );
    //exit(0);

}

void AppMolecularEditorOCL::draw(){
    glClearColor( 0.5f, 0.5f, 0.5f, 1.0f );
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	//glTranslatef( 0.0, 0.0, -5.0 );
	glColor3f( 0.0f,0.0f,0.0f );
	//if(isoOgl)
	//Draw3D::drawAxis(10);

    ray0 = (Vec3d)(cam.pos + cam.rot.a*mouse_begin_x + cam.rot.b*mouse_begin_y);

    /*
    Vec3f p; 
    Vec3f p0    = (Vec3f){ 1.0, -0.5, 0.3 };
    Vec3f force = (Vec3f){ 1.0, 0.3, 2.0 };
    
    Mat3f mrot; qrot.toMatrix(mrot);
    //Mat3f mrot; qrot.toMatrix_T(mrot);
    //mrot.dot_to( p0, p );
    //mrot.dot_to_T( p0, p );
    //qrot.transformVec( p0, p );
    qrot.untransformVec( p0, p );
        
    force = (Vec3f)getForceSpringRay( (Vec3d)p, (Vec3d)cam.rot.c, ray0, -1.0 );
   
    Vec3f torq; torq.set_cross( p, force );
    //Vec3f torq; torq.set_cross( force, p );
    qrot.dRot_exact( 0.01, torq );
    
    //printf( " c.f %g t.f %g t.c %g \n", cam.rot.c.dot(force),   torq.dot(force),   torq.dot(cam.rot.c)/sqrt( torq.norm2()* cam.rot.c.norm2()  )   );
    
    glColor3f(0.0,0.0,0.0); Draw3D::drawVec       ( p );
    glColor3f(0.0,0.0,0.0); Draw3D::drawPointCross( p, 0.05 );
    glColor3f(0.0,1.0,0.0); Draw3D::drawVec       ( torq );
    glColor3f(1.0,0.0,0.0); Draw3D::drawVecInPos  ( force, p );
    
    Draw3D::drawAxis(1.0);
    return;
    */


	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	viewSubstrate( 2, 2, isoOgl, world.gridFF.grid.cell.a, world.gridFF.grid.cell.b );

    
    Draw3D::drawPointCross( ray0, 0.1 );
    //Draw3D::drawVecInPos( camMat.c, ray0 );
    if(ipicked>=0) Draw3D::drawLine( world.apos[ipicked], ray0);

	printf( " # =========== frame %i \n", frameCount );
	
	float dt = 0.5;
	int isystem = 0;
	//*(Vec3f*)(clworld.poses+isystem*clworld.nMols+itest) = (Vec3f)cursor3D;
	
	clworld.system2atoms( isystem, atoms_tmp );
	clworld.evalForceCPU( isystem, world.gridFF, atoms_tmp, fatoms_tmp );
	clworld.moveSystemGD( isystem, dt, 1.0, 1.0 );
	
	//drawAtomsF8(atom_count, atoms_tmp, 0.25, ogl_sph);
	
	glColor3f(0.0,0.0,0.0); drawRigidMolSystem( clworld, isystem );
	drawRigidMolSystemForceTorq(  clworld, isystem, 100.0, 1000.0 );
	glColor3f(1.0,0.0,1.0); drawAtomsForces( atom_count, atoms_tmp, fatoms_tmp, 0.0, 100.0 );
   
	//return;
	
	world.cleanAtomForce();
	world.frags2atoms(); 
	world.eval_FFgrid();
	
	Quat4f*  ps = (Quat4f*)clworld.poses;
	Quat4d* wps = (Quat4d*)  world.poses;
	printf( "  world pose p(%5.5e,%5.5e,%5.5e) q(%5.5e,%5.5e,%5.5e,%5.5e) \n",  wps[0].f.x, wps[0].f.y, wps[0].f.z,  wps[1].x, wps[1].y, wps[1].z, wps[1].w );
	printf( "clworld pose p(%5.5e,%5.5e,%5.5e) q(%5.5e,%5.5e,%5.5e,%5.5e) \n",   ps[0].f.x,  ps[0].f.y,  ps[0].f.z,   ps[1].x,  ps[1].y,  ps[1].z,  ps[1].w );
	for( int ia=0; ia<world.natoms; ia++ ){
        Vec3d& p  =world.apos  [ia];
        Vec3d& f  =world.aforce[ia];
        Vec3d& req=world.aREQ  [ia];
        //printf( "world %i p(%g,%g,%g) f(%g,%g,%g) req(%g,%g,%g) \n", ia, p.x,p.y,p.z,  f.x,f.y,f.z,  req.x,req.y,req.z );
        printf( "  world %i p(%5.5e,%5.5e,%5.5e) f(%5.5e,%5.5e,%5.5e) req(%5.5e,%5.5e,%5.5e) \n", ia, p.x,p.y,p.z,  f.x,f.y,f.z,  req.x,req.y,req.z );
        
        Vec3f& p_  =*(Vec3f*)(atoms_tmp +ia);
        Vec3f& f_  =         fatoms_tmp[ia];
        Vec3f& req_=*(Vec3f*)(((float*)(atoms_tmp+ia))+4); 
        printf( "clworld %i p(%5.5e,%5.5e,%5.5e) f(%5.5e,%5.5e,%5.5e) req(%5.5e,%5.5e,%5.5e) \n", ia, p_.x,p_.y,p_.z,  f_.x,f_.y,f_.z,  req_.x,req_.y,req_.z );
	}
	
	//exit(0);
	
	drawCPU();

};

void AppMolecularEditorOCL::stepCPU( double& F2, bool randomConf ){
    world.cleanAtomForce();
    
    if( randomConf ){
        Vec3d d=(Vec3d){1.0,1.0,1.0};
        Vec3d shift = world.Collision_box.genRandomSample();
        Quat4d qrot;  qrot.fromUniformS3( {randf(),randf(),randf()} );
        world.tryFragPose( 0, false, shift, qrot );
    }

    world.frags2atoms();       //printf( "DEBUG 5.2\n" );
    world.eval_FFgrid();
    world.eval_MorseQ_On2_fragAware();

    if(ipicked>=0){
        Vec3d f = getForceSpringRay( world.apos[ipicked], (Vec3d)cam.rot.c, ray0, -1.0 );
        world.aforce[ipicked].add( f );
    }

    world.cleanPoseTemps();
    world.aforce2frags();      //printf( "DEBUG 5.4\n" );

    for(int i=0; i<world.natoms; i++ ){ Draw3D::drawVecInPos( world.aforce[i]*10.0, world.apos[i] ); }

    world.toDym(true);
    F2 = opt.move_FIRE();  //printf( "DEBUG 5.5\n" );
    world.checkPoseUnitary();
    world.fromDym();  
}

void AppMolecularEditorOCL::drawCPU(){
   
   	double F2;
	perFrame = 1;
	//delay = 100;
	for(int itr=0; itr<perFrame; itr++){
        stepCPU( F2, false );
    }
    
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
    glShadeModel(GL_SMOOTH);
    for(int i=0; i<world.natoms; i++){
        glEnable(GL_LIGHTING);
        Mat3d mat;
        mat.setOne();
        mat.mul( atomSize*params.atypes[world.atypes[i]].RvdW );
        Draw::setRGB( params.atypes[world.atypes[i]].color );
        Draw3D::drawShape(world.apos[i],mat,ogl_sph);
        glDisable(GL_LIGHTING);
    }
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

}

void  AppMolecularEditorOCL::keyStateHandling( const Uint8 *keys ){
    double dstep=0.025;

    if( keys[ SDL_SCANCODE_W ] ){ cursor3D.y -=dstep; }
    if( keys[ SDL_SCANCODE_S ] ){ cursor3D.y +=dstep; }
    if( keys[ SDL_SCANCODE_A ] ){ cursor3D.x -=dstep; }
    if( keys[ SDL_SCANCODE_D ] ){ cursor3D.x +=dstep; }
    if( keys[ SDL_SCANCODE_Q ] ){ cursor3D.z -=dstep; }
    if( keys[ SDL_SCANCODE_E ] ){ cursor3D.z +=dstep; }

    if( keys[ SDL_SCANCODE_X ] ){ cam.pos.z +=0.1; }
    if( keys[ SDL_SCANCODE_Z ] ){ cam.pos.z -=0.1; }

    //AppSDL2OGL_3D::keyStateHandling( keys );
};

void AppMolecularEditorOCL::eventHandling ( const SDL_Event& event  ){
    //printf( "NonInert_seats::eventHandling() \n" );
    switch( event.type ){
        case SDL_KEYDOWN :
            switch( event.key.keysym.sym ){
                //case SDLK_p:  first_person = !first_person; break;
                //case SDLK_o:  perspective  = !perspective; break;
                //case SDLK_r:  world.fireProjectile( warrior1 ); break;

                case SDLK_v: for(int i=0; i<world.natoms; i++){ ((Vec3d*)opt.vel)[i].add(randf(-drndv,drndv),randf(-drndv,drndv),randf(-drndv,drndv)); } break;
                case SDLK_p: for(int i=0; i<world.natoms; i++){ world.apos[i].add(randf(-drndp,drndp),randf(-drndp,drndp),randf(-drndp,drndp)); } break;

                //case SDLK_LEFTBRACKET:  if(ibpicked>=0) world.bond_0[ibpicked] += 0.1; break;
                //case SDLK_RIGHTBRACKET: if(ibpicked>=0) world.bond_0[ibpicked] -= 0.1; break;

                //case SDLK_RIGHTBRACKET: itest++; if(itest>=atomdist.natoms)itest=0;  printf("itest %i\n",itest); break;
                //case SDLK_LEFTBRACKET:  itest--; if(itest<0)itest=atomdist.natoms-1; printf("itest %i\n",itest); break;   
                
                
                case SDLK_RIGHTBRACKET: itest++; if(itest>=clworld.nMols)itest=0;  printf("itest %i\n",itest); break;
                case SDLK_LEFTBRACKET:  itest--; if(itest<0)itest=clworld.nMols-1; printf("itest %i\n",itest); break;   
                             
                
                case SDLK_n: 
                    isystem++; if(isystem>=clworld.nSystems)isystem=0; 
                    atom_count = clworld.system2atoms( isystem, atoms_tmp );
                    break;
                    
                case SDLK_KP_4: qCamera=Quat4fLeft;    printf("cam Left   \n"); break;
                case SDLK_KP_6: qCamera=Quat4fRight;   printf("cam Right  \n"); break;
                case SDLK_KP_5: qCamera=Quat4fBack;    printf("cam Back   \n"); break;
                case SDLK_KP_8: qCamera=Quat4fFront;   printf("cam Front  \n"); break;
                case SDLK_KP_7: qCamera=Quat4fTop;     printf("cam Top    \n"); break;
                case SDLK_KP_9: qCamera=Quat4fBotton;  printf("cam Botton \n"); break;
                
            }
            break;
        case SDL_MOUSEBUTTONDOWN:
            switch( event.button.button ){
                case SDL_BUTTON_LEFT:
                    ipicked = pickParticle( world.natoms, world.apos, ray0, (Vec3d)cam.rot.c , 0.5 );
                    printf("ipicked %i \n", ipicked);
                    break;
                case SDL_BUTTON_RIGHT:
                    ibpicked = world.pickBond( ray0, (Vec3d)cam.rot.c , 0.5 );
                    printf("ibpicked %i \n", ibpicked);
                    break;
            }
            break;
        case SDL_MOUSEBUTTONUP:
            switch( event.button.button ){
                case SDL_BUTTON_LEFT:
                    ipicked = -1;
                    break;
                case SDL_BUTTON_RIGHT:
                    //ibpicked = -1;
                    break;
            }
            break;
    };
    AppSDL2OGL::eventHandling( event );
}

void AppMolecularEditorOCL::drawHUD(){
    glDisable ( GL_LIGHTING );

}

// ===================== MAIN

AppMolecularEditorOCL * thisApp;

int main(int argc, char *argv[]){
	SDL_Init(SDL_INIT_VIDEO);
	SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
	//SDL_SetRelativeMouseMode( SDL_TRUE );
	int junk;
	thisApp = new AppMolecularEditorOCL( junk , 800, 600 );
	thisApp->loop( 1000000 );
	return 0;
}
















