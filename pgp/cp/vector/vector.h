#ifndef _VECTOR_H_
#define _VECTOR_H_

namespace Vector {

    struct TVector3 {
        double x;
        double y;
        double z;
    };

    double Length(TVector3 v);
    double Dot(TVector3 a, TVector3 b);
	TVector3 Prod(TVector3 a, TVector3 b);
	TVector3 Normalize(TVector3 v);
	TVector3 Sub(TVector3 a, TVector3 b);
	TVector3 Add(TVector3 a, TVector3 b);
	TVector3 Mult(TVector3 a, TVector3 b, TVector3 c, TVector3 v);
    TVector3 Mult(double coef, TVector3 v);
}

#endif // _VECTOR_H_
