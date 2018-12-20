
#ifndef  SphereTreeND_h
#define  SphereTreeND_h

#include <vector>
#include <unordered_set>

#include "VecN.h"

/*
 Note : point is added to only one sphereNode, therefore
 findNeighs( double * vec, double R, std::unordered_set<int>* found )
 does not have to use std::unordered_set but it can use just vector or buffer array without worry of reperition

 Consideration 1 : if samples are generated by something like random-walk, it would be reasonable first check neighboring sphere nodes, therefore building connection list

 Consideration 2 : would it be worse or better to center the spheres in integer grid points? what would be than main differecne form ND-cubic grid based hashmap ?

 Consideration 3 :  building n-dimensional hash map is not problem ( can do string hashes ), but problem is large number of neighbors of each cell in nD

 Consideration 4 : How this is connected to neural networks ? Basically we can online create and kill radial basis function in areas where is lot of samples.
                   This means building neural-network with dynamic topology; Can it be related to grow of synapses/neurons in biological neural nets ?

*/


// ================ Algorithm based on recursion



//bulshit !



inline double dist2limited( int n, double * xs, double * x0s, double dist2max ){
    double d2sum = 0;
    for(int i=0; i<n; i++){ double d = d-x0s[i]; d*=d; d2sum+=d; }
    //for(int i=0; i<n; i++){ double d=center[i]-d; d*=d; d2sum+=d;  if(d2sum>dist2max) return 1e+300; }
    return d2sum;
}

/*
class SphereNodeND_Abstract{
    public:
    double * center;
    virtual  SphereNodeND_Abstract* findNearest( int n, double xs, double dist2max, double& d, int& level );
}

class SphereNodeND_Branch{
    public:
    std::vector<SphereNodeND_Abstract*> branches;

    virtual SphereNodeND_Abstract* findNearest( int n, double xs, double dist2max, double& d, int& level ){
        for( sph : branches ){
            double r2 = dist2limited( nDim, vec, sph->center, dist2max );
            if (r2>R2OUT) continue;
            findNearest( n, xs, dist2max, d, level );
        }
    };
}

class SphereNodeND_Leaf{
    public:
    std::vector<double*> leafs;

    virtual SphereNodeND_Abstract* findNearest( int n, double xs, double dist2max, double& d, int& level ){
        if(level) ;
    };
}

class SphereNodeND_Root{

}
*/




//=================================================
//================================================= Other version
//=================================================

class SphereNodeND{
    public:
    double * center;
    //double R;
    //std::vector<SphereNodeND*> branches;  // later we should make multi-level branching
    std::vector<int>    leafs; // int can be both branch and leaf

    // try shift sphere center toward 'vec' by least amout that sphere contains vec, check if all leafs are still inside sthis sphereNode
    inline bool tryShift( int n, double * vec, double R ){};

};

class SphereTreeND{
    public:
    int     nDim;
    double  R_contain;   // if |p-sph.center|<R_contain point        'p' is child of sph
    //double  R_overlap; // if |p-sph.center|<R_overlap neighbors of 'p' should searched in sph

    std::vector<double*>       points;
    std::vector<SphereNodeND*> branches;

    SphereNodeND* findClosestBranch( double * vec, double dist2max, double& d2min ){
        d2min                 = 1e+300;
        SphereNodeND* sph_min = NULL;
        for(SphereNodeND* sph: branches ){
            double r2 = dist2limited( nDim, vec, sph->center, dist2max );
            if( r2 > dist2max ) continue;
            if( r2 < d2min    ){ sph_min = sph;  d2min=r2; }
        }
        return sph_min;
    }

    SphereNodeND* findNeighs( double * vec, double R, std::unordered_set<int>* found ){
        double R2        = R*R;
        double R_overlap = R + R_contain;
        double dist2max  = R_overlap*R_overlap;
        double d2min     = 1e+300;
        SphereNodeND* sph_min = NULL;
        for(SphereNodeND* sph: branches ){
            //double dist2i = sph->dist2( nDim, vec, dist2max );
            double dist2i = dist2limited( nDim, vec, sph->center, dist2max  );
            if( dist2i > dist2max ) continue;
            if( dist2i < d2min    ){ sph_min = sph;  d2min=dist2i; }
            if(!found) continue;
            for( int id : sph->leafs ){
                double d2id = VecN::err2( nDim, vec, points[id] );
                if( d2id<R2 ) found->insert(id);  // this could be quite slow operation
            }
        }
        return sph_min;
    }

    bool insert( double * vec, int id, SphereNodeND* sph_min, double d2min ){
        //double dist2max = R_contain*R_contain;
        if( d2min > (R_contain*R_contain) ){ // new node required ?
            // tryShift( int n, double * vec, double R )  // posibly in future ?
            sph_min = new SphereNodeND();
            sph_min->center = vec;
            branches.push_back(sph_min);
        }
        sph_min->leafs.push_back(id);
    };

    void insert( double * vec, int id ){
        double dist2max = R_contain*R_contain;
        double d2min = 1e+300;
        SphereNodeND* sph_min = NULL;
        //for( int i=0; branches.size(); i++ ){
        for(SphereNodeND* sph: branches ){
            //SphereNodeND* sph = branches[i];
            //double dist2i = sph->dist2( nDim, vec, dist2max );
            double dist2i = dist2limited( nDim, vec, sph->center, dist2max  );
            //if( dist2i > dist2max ) continue;
            if( dist2i < d2min    ){ sph_min = sph;  d2min=dist2i; }
        }
        insert( vec, id, sph_min, d2min );
    }

    // insert only if does not overlap Rmin distance of some other sphere
    void insert( double * vec, int id, double Rmin ){

    }

};

#endif

