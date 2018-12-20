

// read also:
// http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/

#ifndef  quaternion_h
#define  quaternion_h

#include <math.h>
#include <cstdlib>
#include <stdio.h>

#include "fastmath.h"
#include "Vec3.h"
#include "Mat3.h"

template <class TYPE>
inline TYPE project_beam_to_sphere( TYPE r, TYPE x, TYPE y ){
	TYPE z;
	TYPE r2 = r * r;
	TYPE d2 = x*x + y*y;
	if ( d2 < ( 0.5d * r2 ) ) {
		z = sqrt( r2 - d2 );
	} else {
		TYPE t2 = 0.5d * r;
		z = sqrt( t2 / d2 );
	}
	return z;
}

/*
float project_beam_to_sphere( float r, float x, float y ){
	float d, t, z;
	d = sqrt( x*x + y*y );
	if ( d < r * 0.70710678118654752440 ) {
		z = sqrt( r*r - d*d );
	} else {
		t = r * 0.70710678118654752440; // == 1/sqrt(2)
		z = t*t / d;
	}
	return z;
}
*/

//template <class TYPE, class VEC, class MAT, class QUAT>
//template <class TYPE, class VEC>
template <class TYPE>
class Quat4TYPE {
	using VEC  = Vec3TYPE<TYPE>;
	using MAT  = Mat3TYPE<TYPE>;
	using QUAT = Quat4TYPE<TYPE>;
	public:
	union{
		struct{ TYPE x,y,z,w; };
		TYPE array[4];
	};

    inline void set   ( TYPE f                             ){ x=f;   y=f;   z=f;   w=f;   };
	inline void set   ( const  QUAT& q                     ){ x=q.x; y=q.y; z=q.z; w=q.w; };
	inline void set   ( TYPE fx, TYPE fy, TYPE fz, TYPE fw ){ x=fx;  y=fy;  z=fz;  w=fw;  };
	inline void setOne( ){ x=y=z=0; w=1; };

// ====== basic aritmetic

// ============== Basic Math

    inline void mul        ( TYPE f                               ){ x*=f;            y*=f;           z*=f;           w*=f;            };
    inline void add        ( const QUAT& v                        ){ x+=v.x;          y+=v.y;         z+=v.z;         w+=v.w;          };
    inline void sub        ( const QUAT& v                        ){ x-=v.x;          y-=v.y;         z-=v.z;         w-=v.w;          };
	inline void set_add    ( const QUAT& a, const QUAT& b         ){ x =a.x+b.x;      y =a.y+b.y;     z =a.z+b.z;     w =a.w+b.w;      };
	inline void set_sub    ( const QUAT& a, const QUAT& b         ){ x =a.x-b.x;      y =a.y-b.y;     z =a.z-b.z;     w =a.w-b.w;      };
	inline void add_mul    ( const QUAT& a, TYPE f                ){ x+=a.x*f;        y+=a.y*f;       z+=a.z*f;       w+=a.w*f;        };
	inline void set_add_mul( const QUAT& a, const QUAT& b, TYPE f ){ x =a.x + f*b.x;  y =a.y + f*b.y; z =a.z + f*b.z; w =a.w + f*b.w;  };

    inline TYPE dot  ( QUAT q   ) {  return       w*q.w + x*q.x + y*q.y + z*q.z;   }
    inline TYPE norm2(          ) {  return       w*  w + x*  x + y*  y + z*  z;   }
	inline TYPE norm (          ) {  return sqrt( w*  w + x*  x + y*  y + z*  z ); }
    inline TYPE normalize() {
		TYPE norm  = sqrt( x*x + y*y + z*z + w*w );
		TYPE inorm = 1.0d/norm;
		x *= inorm;    y *= inorm;    z *= inorm;   w *= inorm;
		return norm;
    }

// ====== Quaternion multiplication

	inline void setQmul( const QUAT& a, const QUAT& b) {
        x =  a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x;
        y = -a.x * b.z + a.y * b.w + a.z * b.x + a.w * b.y;
        z =  a.x * b.y - a.y * b.x + a.z * b.w + a.w * b.z;
        w = -a.x * b.x - a.y * b.y - a.z * b.z + a.w * b.w;
    };

    inline void qmul( const QUAT& a) {
        TYPE aw = a.w, ax = a.x, ay = a.y, az = a.z;
        TYPE x_ =  x * aw + y * az - z * ay + w * ax;
        TYPE y_ = -x * az + y * aw + z * ax + w * ay;
        TYPE z_ =  x * ay - y * ax + z * aw + w * az;
                w = -x * ax - y * ay - z * az + w * aw;
        x = x_; y = y_; z = z_;
    };

    inline void qmul_T( const QUAT& a) {
        TYPE aw = a.w, ax = a.x, ay = a.y, az = a.z;
        TYPE x_ =  ax * w + ay * z - az * y + aw * x;
        TYPE y_ = -ax * z + ay * w + az * x + aw * y;
        TYPE z_ =  ax * y - ay * x + az * w + aw * z;
              w = -ax * x - ay * y - az * z + aw * w;
        x = x_; y = y_; z = z_;
    };

    inline void invertUnitary() { x=-x; y=-y; z=-z; }

    inline void invert() {
		TYPE norm = sqrt( x*x + y*y + z*z + w*w );
		if ( norm > 0.0 ) {
			TYPE invNorm = 1.0d / norm;
			x *= -invNorm; y *= -invNorm;z *= -invNorm;	w *=  invNorm;
		}
    };



// ======= Conversion : Angle & Axis

	inline void fromAngleAxis( TYPE angle, const VEC& axis ){
		TYPE ir   = 1/axis.norm();
		VEC  hat  = axis * ir;
		TYPE a    = 0.5d * angle;
		TYPE sa   = sin(a);
		w =           cos(a);
		x = sa * hat.x;
		y = sa * hat.y;
		z = sa * hat.z;
	};

	void fromCosAngleAxis( TYPE scal_prod, const VEC& axis ){
	// we assume -phi instead of phi!!!, minus effectively implies sa -> -s
		constexpr TYPE cos_cutoff = 1 - 1e-6;
		TYPE cosphi, sinphi, sa, phi, sgn_sinphi;
		TYPE ir    = 1.0 / axis.norm();
		VEC  hat   = axis * ir;
		cosphi     = scal_prod;
		sgn_sinphi = 1.0; // ?
		if( cosphi > cos_cutoff ){
			sa = 0; w = 1;
		} else if( cosphi < -( cos_cutoff ) ){
			sa = -1; w = 0;
		} else {
			sa = + sqrt( (1 - cosphi) / 2.0 );
			w  = - sqrt( (1 + cosphi) / 2.0 ) * sgn_sinphi;
//			sa = -sa; w = -w;
		}
		x = sa * hat.x;
		y = sa * hat.y;
		z = sa * hat.z;

	}

	#define TRACKBALLSIZE ( 0.8 )

	/*
	void fromTrackball( TYPE p1x, TYPE p1y, TYPE p2x, TYPE p2y ){
		VEC  axis; // axis of rotation
		//TYPE phi;  // angle of rotation
		VEC  p1, p2, d;
		//TYPE t;
		//if( ( sq(p2x-p1x)+sq(p2y-p1y) ) < 1e-8 ){   }
		if( ( p2x == p1x ) && ( p2y == p1y ) ){ setOne(); return; }
		p1.set( p1x, p1y, project_beam_to_sphere<TYPE>( TRACKBALLSIZE, p1x, p1y ) );
		p2.set( p2x, p2y, project_beam_to_sphere<TYPE>( TRACKBALLSIZE, p2x, p2y ) );
		axis.set_cross( p2, p1 );


		TYPE t = d.norm() / ( 2.0 * TRACKBALLSIZE );
		if( t > 1.0 )  t =  1.0;
		if( t < -1.0 ) t = -1.0;
        TYPE phi = 2.0 * asin( t );
        fromAngleAxis( phi, axis );


        //TYPE t = sqrt( 1 - d.norm2() );
        //fromCosAngleAxis( t, axis );

		// TYPE cosphi = ;
		// phi = 2.0 * asin( t );
		// fromCosAngleAxis( cosphi, axis );

	}
	*/

	void fromTrackball( TYPE p1x, TYPE p1y, TYPE p2x, TYPE p2y ){
		VEC  axis; // axis of rotation
		//TYPE phi;  // angle of rotation
		VEC  p1, p2, d;
		//TYPE t;
		//if( p1x == p2x && p1y == p2y ){	setOne(); return; }
		if( ( sq<TYPE>(p2x-p1x)+sq<TYPE>(p2y-p1y) ) < 1e-8 ){ setOne(); return; }
		p1.set( p1x, p1y, project_beam_to_sphere<TYPE>( TRACKBALLSIZE, p1x, p1y ) );
		p2.set( p2x, p2y, project_beam_to_sphere<TYPE>( TRACKBALLSIZE, p2x, p2y ) );
		axis.set_cross( p2, p1 );
		d.set_sub( p1, p2 );

        /*
		TYPE t = d.norm() / ( 2.0 * TRACKBALLSIZE );
		if( t > 1.0 )  t =  1.0;
		if( t < -1.0 ) t = -1.0;
		TYPE phi = 2.0 * asin( t );
		fromAngleAxis( phi, axis );
		*/

        TYPE t = sqrt( 1 - d.norm2() );
        fromCosAngleAxis( t, axis );

	}


/*
    void fromTrackball_0( TYPE px, TYPE py ){
		VEC  axis; // axis of rotation
		TYPE phi;  // angle of rotation
		VEC  p1, p2, d;
		TYPE t;
		if( ( px == 0 ) && ( p == py ) ){
			setOne();
			return;
		}
		p2.set( px, py, project_beam_to_sphere<TYPE>( TRACKBALLSIZE, px, py ) );
		p1.set( 0, 0, TRACKBALLSIZE );
		axis.set_cross( p2, p1 );
		d.set_sub( p1, p2 );
		t = d.norm() / ( 2.0 * TRACKBALLSIZE );
		if( t > 1.0 )  t =  1.0;
		if( t < -1.0 ) t = -1.0;

		phi = 2.0 * asin( t );
		fromAngleAxis( phi, axis );

	}
*/

// =======  pitch, yaw, roll

	inline void dpitch( TYPE angle ){ TYPE ca,sa; sincos_taylor2(angle,sa,ca); pitch( ca, sa );  };
	inline void pitch ( TYPE angle ){ pitch( cos(angle), sin(angle) );  };
    inline void pitch ( TYPE ca, TYPE sa ) {
        TYPE x_ =  x * ca + w * sa;
        TYPE y_ =  y * ca + z * sa;
        TYPE z_ = -y * sa + z * ca;
                w = -x * sa + w * ca;
        x = x_; y = y_; z = z_;
    };

	inline void dyaw( TYPE angle ){ TYPE ca,sa; sincos_taylor2(angle,sa,ca); yaw( ca, sa );  };
	inline void yaw ( TYPE angle ){ yaw( cos(angle), sin(angle) );  };
    inline void yaw ( TYPE ca, TYPE sa ) {
        TYPE x_ =  x * ca - z * sa;
        TYPE y_ =  y * ca + w * sa;
        TYPE z_ =  x * sa + z * ca;
                w = -y * sa + w * ca;
        x = x_; y = y_; z = z_;
    };

	inline void droll( TYPE angle ){ TYPE ca,sa; sincos_taylor2(angle,sa,ca); roll( ca, sa );  };
	inline void roll ( TYPE angle ){ roll( cos(angle), sin(angle) );  };
    inline void roll ( TYPE ca, TYPE sa ) {
        TYPE x_ =  x * ca + y * sa;
        TYPE y_ = -x * sa + y * ca;
        TYPE z_ =  z * ca + w * sa;
                w = -z * sa + w * ca;
        x = x_; y = y_; z = z_;
    };


	inline void dpitch2( TYPE angle ){ TYPE ca,sa; sincos_taylor2(angle,sa,ca); pitch2( ca, sa );  };
	inline void pitch2 ( TYPE angle ){ pitch2( cos(angle), sin(angle) );  };
    inline void pitch2 ( TYPE ca, TYPE sa ) {
/*
        TYPE x_ =  ax * w + ay * z - az * y + aw * x;
        TYPE y_ = -ax * z + ay * w + az * x + aw * y;
        TYPE z_ =  ax * y - ay * x + az * w + aw * z;
               w  = -ax * x - ay * y - az * z + aw * w;
*/
        TYPE x_ =  sa * w + ca * x;
        TYPE y_ = -sa * z + ca * y;
        TYPE z_ =  sa * y + ca * z;
               w  = -sa * x + ca * w;
        x = x_; y = y_; z = z_;
    };

	inline void dyaw2( TYPE angle ){ TYPE ca,sa; sincos_taylor2(angle,sa,ca); yaw2( ca, sa );  };
	inline void yaw2 ( TYPE angle ){ yaw2( cos(angle), sin(angle) );  };
    inline void yaw2 ( TYPE ca, TYPE sa ) {
        TYPE x_ = + sa * z  + ca * x;
        TYPE y_ = + sa * w  + ca * y;
        TYPE z_ = - sa * x  + ca * z;
               w  = - sa * y  + ca * w;
        x = x_; y = y_; z = z_;
    };

	inline void droll2( TYPE angle ){ TYPE ca,sa; sincos_taylor2(angle,sa,ca); roll2( ca, sa );  };
	inline void roll2 ( TYPE angle ){ roll2( cos(angle), sin(angle) );  };
    inline void roll2 ( TYPE ca, TYPE sa ) {
        TYPE x_ = - sa * y + ca * x;
        TYPE y_ = + sa * x + ca * y;
        TYPE z_ = + sa * w + ca * z;
               w  = - sa * z + ca * w;
        x = x_; y = y_; z = z_;
    };

// ====== Differential rotation

	inline void dRot_exact ( TYPE dt, const VEC& omega ) {
		TYPE hx   = omega.x;
		TYPE hy   = omega.y;
		TYPE hz   = omega.z;
		TYPE r2   = hx*hx + hy*hy + hz*hz;
		if(r2>0){
			TYPE norm = sqrt( r2 );
			TYPE a    = dt * norm * 0.5d;
			TYPE sa   = sin( a )/norm;  // we normalize it here to save multiplications
			TYPE ca   = cos( a );
			hx*=sa; hy*=sa; hz*=sa;            // hat * sin(a)
			TYPE x_ = x, y_ = y, z_ = z, w_ = w;
			x =  hx*w_ + hy*z_ - hz*y_ + ca*x_;
			y = -hx*z_ + hy*w_ + hz*x_ + ca*y_;
			z =  hx*y_ - hy*x_ + hz*w_ + ca*z_;
			w = -hx*x_ - hy*y_ - hz*z_ + ca*w_;
		}
	};


	inline void dRot_taylor2 ( TYPE dt, VEC& omega ) {
		TYPE hx   = omega.x;
		TYPE hy   = omega.y;
		TYPE hz   = omega.z;
		TYPE r2   = hx*hx + hy*hy + hz*hz;
		TYPE b2   = dt*dt*r2;
		const TYPE c2 = 1.0d/8;    // 4  *2
		const TYPE c3 = 1.0d/48;   // 8  *2*3
		const TYPE c4 = 1.0d/384;  // 16 *2*3*4
		const TYPE c5 = 1.0d/3840; // 32 *2*3*4*5
		TYPE sa   = dt * ( 0.5d - b2*( c3 - c5*b2 ) );
		TYPE ca   =      ( 1    - b2*( c2 - c4*b2 ) );
		hx*=sa; hy*=sa; hz*=sa;  // hat * sin(a)
		TYPE x_ = x, y_ = y, z_ = z, w_ = w;
		x =  hx*w_ + hy*z_ - hz*y_ + ca*x_;
		y = -hx*z_ + hy*w_ + hz*x_ + ca*y_;
		z =  hx*y_ - hy*x_ + hz*w_ + ca*z_;
		w = -hx*x_ - hy*y_ - hz*z_ + ca*w_;
	};

	//  QUAT -> VEC

    inline void rotate(VEC& v){
        // rotate vector by quternion
        // should be optimized later
        // http://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
        QUAT qv; qv.set(v.x,v.y,v.z);
        qmul  ( qv);
        qmul_T( qv );
        v.set(qv.x,qv.y,qv.z);
    }

    inline void dirFw  ( VEC& result)  const  {
        result.zx =     2 * ( x*z - y*w );
		result.zy =     2 * ( y*z + x*w );
		result.zz = 1 - 2 * ( x*x + y*y );
    }

    inline void dirUp  ( VEC& result)  const  {
        result.x =     2 * ( x*y + z*w );
		result.y = 1 - 2 * ( x*x + z*z );
		result.z =     2 * ( y*z - x*w );
    }

    inline void dirLeft( VEC& result)  const  {
        result.xx = 1 - 2 * ( y*y + z*z );
		result.xy =     2 * ( x*y - z*w );
		result.xz =     2 * ( x*z + y*w );
    }


	inline void toMatrix( MAT& result) const {
		    TYPE r2 = w*w + x*x + y*y + z*z;
		    //TYPE s  = (r2 > 0) ? 2d / r2 : 0;
			TYPE s  = 2 / r2;
		    // compute xs/ys/zs first to save 6 multiplications, since xs/ys/zs
		    // will be used 2-4 times each.
		    TYPE xs = x * s;  TYPE ys = y * s;  TYPE zs = z * s;
		    TYPE xx = x * xs; TYPE xy = x * ys; TYPE xz = x * zs;
		    TYPE xw = w * xs; TYPE yy = y * ys; TYPE yz = y * zs;
		    TYPE yw = w * ys; TYPE zz = z * zs; TYPE zw = w * zs;
		    // using s=2/norm (instead of 1/norm) saves 9 multiplications by 2 here
		    result.xx = 1 - (yy + zz);
		    result.xy =     (xy - zw);
		    result.xz =     (xz + yw);
		    result.yx =     (xy + zw);
		    result.yy = 1 - (xx + zz);
		    result.yz =     (yz - xw);
		    result.zx =     (xz - yw);
		    result.zy =     (yz + xw);
		    result.zz = 1 - (xx + yy);
	};


    inline void toMatrix_T( MAT& result) const {
		    TYPE r2 = w*w + x*x + y*y + z*z;
		    //TYPE s  = (r2 > 0) ? 2d / r2 : 0;
			TYPE s  = 2 / r2;
		    // compute xs/ys/zs first to save 6 multiplications, since xs/ys/zs
		    // will be used 2-4 times each.
		    TYPE xs = x * s;  TYPE ys = y * s;  TYPE zs = z * s;
		    TYPE xx = x * xs; TYPE xy = x * ys; TYPE xz = x * zs;
		    TYPE xw = w * xs; TYPE yy = y * ys; TYPE yz = y * zs;
		    TYPE yw = w * ys; TYPE zz = z * zs; TYPE zw = w * zs;
		    // using s=2/norm (instead of 1/norm) saves 9 multiplications by 2 here
		    result.xx = 1 - (yy + zz);
		    result.yx =     (xy - zw);
		    result.zx =     (xz + yw);
		    result.xy =     (xy + zw);
		    result.yy = 1 - (xx + zz);
		    result.zy =     (yz - xw);
		    result.xz =     (xz - yw);
		    result.yz =     (yz + xw);
		    result.zz = 1 - (xx + yy);
	};


	inline void toMatrix_unitary( MAT& result)  const  {
		TYPE xx = x * x;
		TYPE xy = x * y;
		TYPE xz = x * z;
		TYPE xw = x * w;
		TYPE yy = y * y;
		TYPE yz = y * z;
		TYPE yw = y * w;
		TYPE zz = z * z;
		TYPE zw = z * w;
		result.xx = 1 - 2 * ( yy + zz );
		result.xy =     2 * ( xy - zw );
		result.xz =     2 * ( xz + yw );
		result.yx =     2 * ( xy + zw );
		result.yy = 1 - 2 * ( xx + zz );
		result.yz =     2 * ( yz - xw );
		result.zx =     2 * ( xz - yw );
		result.zy =     2 * ( yz + xw );
		result.zz = 1 - 2 * ( xx + yy );
	};


	inline void toMatrix_unitary2( MAT& result)  const  {
		TYPE x2 = 2*x;
		TYPE y2 = 2*y;
		TYPE z2 = 2*z;
		TYPE xx = x2 * x;
		TYPE xy = x2 * y;
		TYPE xz = x2 * z;
		TYPE xw = x2 * w;
		TYPE yy = y2 * y;
		TYPE yz = y2 * z;
		TYPE yw = y2 * w;
		TYPE zz = z2 * z;
		TYPE zw = z2 * w;
		result.xx = 1 - ( yy + zz );
		result.xy =     ( xy - zw );
		result.xz =     ( xz + yw );
		result.yx =     ( xy + zw );
		result.yy = 1 - ( xx + zz );
		result.yz =     ( yz - xw );
		result.zx =     ( xz - yw );
		result.zy =     ( yz + xw );
		result.zz = 1 - ( xx + yy );
	};

	// this will compute force on quaternion from force on some point in coordinate system of the quaternion
	inline void addForceFromPoint( const VEC& p, const VEC& fp, QUAT& fq ) const {
		// dE/dx = dE/dpx * dpx/dx
		// dE/dx = fx
		TYPE px_x =    p.b*y +             p.c*z;
		TYPE py_x =    p.a*y - 2*p.b*x +   p.c*w;
		TYPE pz_x =    p.a*z -   p.b*w - 2*p.c*x;
		fq.x += 2*( fp.x * px_x  +  fp.y * py_x  + fp.z * pz_x );

		TYPE px_y = -2*p.a*y +   p.b*x -   p.c*w;
		TYPE py_y =    p.a*x +             p.c*z;
		TYPE pz_y =    p.a*w +   p.b*z - 2*p.c*y;
		fq.y += 2*( fp.x * px_y  +  fp.y * py_y  + fp.z * pz_y );

		TYPE px_z = -2*p.a*z +   p.b*w +   p.c*x;
		TYPE py_z =   -p.a*w - 2*p.b*z +   p.c*y;
		TYPE pz_z =    p.a*x +   p.b*y;
		fq.z += 2*( fp.x * px_z  +  fp.y * py_z  + fp.z * pz_z );

		TYPE px_w =    p.b*z -   p.c*y;
		TYPE py_w =   -p.a*z +   p.c*x;
		TYPE pz_w =    p.a*y -   p.b*x;
		fq.w += 2*( fp.x * px_w  +  fp.y * py_w  + fp.z * pz_w );

	}


	inline void fromMatrix( const VEC& a, const VEC& b, const VEC& c ) { fromMatrix( a.x,  a.y,  a.z,  b.x,  b.y,  b.z,  c.x,  c.y,  c.z  );  }
	inline void fromMatrix( const MAT& M                             ) { fromMatrix( M.ax, M.ay, M.az, M.bx, M.by, M.bz, M.cx, M.cy, M.cz );  }
	inline void fromMatrix( TYPE m00, TYPE m01, TYPE m02,    TYPE m10, TYPE m11, TYPE m12,        TYPE m20, TYPE m21, TYPE m22) {
        // Use the Graphics Gems code, from
        // ftp://ftp.cis.upenn.edu/pub/graphics/shoemake/quatut.ps.Z
        TYPE t = m00 + m11 + m22;
        // we protect the division by s by ensuring that s>=1
        if (t >= 0) { // by w
            TYPE s = sqrt(t + 1);
            w = 0.5 * s;
            s = 0.5 / s;
            x = (m21 - m12) * s;
            y = (m02 - m20) * s;
            z = (m10 - m01) * s;
        } else if ((m00 > m11) && (m00 > m22)) { // by x
            TYPE s = sqrt(1 + m00 - m11 - m22);
            x = s * 0.5;
            s = 0.5 / s;
            y = (m10 + m01) * s;
            z = (m02 + m20) * s;
            w = (m21 - m12) * s;
        } else if (m11 > m22) { // by y
            TYPE s = sqrt(1 + m11 - m00 - m22);
            y = s * 0.5;
            s = 0.5 / s;
            x = (m10 + m01) * s;
            z = (m21 + m12) * s;
            w = (m02 - m20) * s;
        } else { // by z
            TYPE s = sqrt(1 + m22 - m00 - m11);
            z = s * 0.5;
            s = 0.5 / s;
            x = (m02 + m20) * s;
            y = (m21 + m12) * s;
            w = (m10 - m01) * s;
        }
	}



};

/*
class Quat4i : public Quat4TYPE< int,    Vec3i, Mat3i, Quat4i >{  };
class Quat4f : public Quat4TYPE< float,  Vec3f, Mat3f, Quat4f >{  };
class QUAT : public Quat4TYPE< TYPE, VEC, MAT, QUAT >{  };
*/

using Quat4i = Quat4TYPE< int>;
using Quat4f = Quat4TYPE< float>;
using Quat4d = Quat4TYPE< double >;


inline void convert( const Quat4f& from, Quat4d& to ){ to.x=from.x;        to.y=from.y;        to.z=from.z;        to.w=from.w;        };
inline void convert( const Quat4d& from, Quat4f& to ){ to.x=(float)from.x; to.y=(float)from.y; to.z=(float)from.z; to.w=(float)from.w; };


#endif

