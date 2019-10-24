#ifndef LORENZ95_H
#define LORENZ95_H

#include <armadillo>
#include <omp.h>
#include <stack>
#include <string>

//#include "funaux.h"

using namespace arma;
using namespace std;

template<class T>
T f95(const T& X){

	int Nc = X.n_cols;
	int Nl = X.n_rows;
	int c;

	T Y(Nl,Nc);
//#pragma omp parallel for
	for (c =0; c<Nc ;c++){
		for (int l=0; l<Nl ; l++){
			Y(l,c) = X((l+1)%Nl,c)*X((l+Nl-1)%Nl,c)-X((l+Nl-1)%Nl,c)*X((l+Nl-2)%Nl,c)-X(l%Nl,c)+8.0;
		}
	}
	return Y;
}

template<class T>
T f63(const T& X){

	unsigned int Nc = X.n_cols;
	unsigned int Nl = 3;
	if (X.n_rows!=Nl){
		cout << "Wrong dimension L63" << endl;
	}

	unsigned int c;

	T Y(Nl,Nc);
	for (c =0; c<Nc ;c++){
		Y(0,c) = 10.0*(X(1,c)-X(0,c));
		Y(1,c) = 28.0*X(0,c)-X(1,c)-X(0,c)*X(2,c);
		Y(2,c) = X(0,c)*X(1,c)-(8.0/3.0)*X(2,c);
	}
	return Y;
}

//Adjoint de f
mat gf95(vec x, mat gy){
    //En avant
    int nl = gy.n_rows;
    int nc = gy.n_cols;
    //Initialisation des variables adjointes
    mat gx(nl,nc,fill::zeros);
    //En arrière
    for (int i=0;i<nl;i++){
        gx.row(i) += - gy.row(i);
        gx.row((i+1)%nl) += x((i+nl-1)%nl)*gy.row(i);
        gx.row((i+nl-1)%nl) += - x((i+nl-2)%nl)*gy.row(i)
                        + gy.row(i)*x((i+1)%nl);
        gx.row((i+nl-2)%nl) += - gy.row(i)*x((i+nl-1)%nl);
    }
    return gx;
}
//Tangent de f
mat df95(vec x, mat dx){
    //Paramètre du modèle
    int nl = dx.n_rows;
    int nc = dx.n_cols;
    mat dy(nl,nc);
    for (int i=0;i<nl;i++){
        dy.row(i) = -dx.row((i+nl-2)%nl)*x((i+nl-1)%nl)-x((i+nl-2)%nl)*dx.row((i+nl-1)%nl)
                +dx.row((i+nl-1)%nl)*x((i+1)%nl)+x((i+nl-1)%nl)*dx.row((i+1)%nl)
				-dx.row(i);
    }
    return dy;
}
template<class T>
T RK(const T& X, string M){

	T (*f)(const T&) = f95;
	double dt = 0.05;
	if(M.compare("L63")==0){
		f = f63;
		dt = 0.01;
	}

	T k1 = f(X);
	T k2 = X+(dt/2.0)*k1;
	k2 = f(k2);
	T k3 = X+(dt/2.0)*k2;
	k3 = f(k3);
	T k4 = X+dt*k3;
	k4 = f(k4);

	return X+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4);
}
mat gRK95(vec x, mat gy, double dt){

	//En avant
	vec v0 = x;
	vec k1 = f95(v0);

	vec v1 = v0+(dt/2.0)*k1;
	vec k2 = f95(v1);

	vec v2 = v0+(dt/2.0)*k2;
	vec k3 = f95(v2);

	vec v3 = v0+dt*k3;
	//k4 = f(v3);
	//vec y = x+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4);

	//Initialisation des variables adjointes
	mat zeros = mat(gy.n_rows,gy.n_cols,fill::zeros);
	mat gk4,gk3,gk2,gk1,gv0,gv3,gv2,gv1,gx;
	gk4=gk3=gk2=gk1=gv0=gv3=gv2=gv1=gx=zeros;
	//En arrière
	gk4 += (dt/6.0)*gy;
	gk3 += (dt/3.0)*gy;
	gk2 += (dt/3.0)*gy;
	gk1 += (dt/6.0)*gy;
	gv0 += gy;
	gy = zeros;

	gv3 += gf95(v3,gk4);
	gk4 = zeros;

	gk3 += dt*gv3;
	gv0 += gv3;
	gv3 = zeros;

	gv2 += gf95(v2,gk3);
	gk3 = zeros;

	gk2 += (dt/2.0)*gv2;
	gv0 += gv2;
	gv2 = zeros;

	gv1 += gf95(v1,gk2);
	gk2 = zeros;

	gk1 += (dt/2.0)*gv1;
    gv0 += gv1;
    gv1 = zeros;

    gv0 += gf95(v0,gk1);
    gk1 = zeros;

    gx += gv0;
    gv0 = zeros;

    return gx;
}
//Tangent de RK
mat dRK95(vec x, mat dx, double dt){
    vec k1 = f95(x);
    mat dk1 = df95(x,dx);
    vec k2 = f95((x+(dt/2.0)*k1).eval());
    mat dk2 = df95((x+(dt/2.0)*k1).eval(),(dx+(dt/2.0)*dk1).eval());
    vec k3 = f95((x+(dt/2.0)*k2).eval());
    mat dk3 = df95((x+(dt/2.0)*k2).eval(),(dx+(dt/2.0)*dk2).eval());
    vec k4 = f95((x+dt*k3).eval());
    mat dk4 = df95((x+dt*k3).eval(),(dx+dt*dk3).eval());
    return dx+(dt/6.0)*(dk1+2.0*dk2+2.0*dk3+dk4);
}
//Résolvante de l'équation différentielle sur N pas de temps
template<class T>
void M(T& X, int N, string M){
	for (int i=0; i<N ; i++){
		X = RK(X,M);
	}
}

//Tangent du modèle
mat dM(vec x, int N, mat dx,string M,double dt){
	for (int i=0; i<N; i++){
		dx = dRK95(x,dx,dt);
		x = RK(x,M);
	}
	return dx;
}
//Adjoint du modèle
mat gM(vec x, int N, mat gy, string M,double dt){
	std::stack<vec> trace;
	//En avant
	for (int i=0; i<N ; i++){
		trace.push(x);
		x = RK(x,M);
	}
	//En arrière
	mat gx = gy;
	for (int i=0; i<N ; i++){
		x = trace.top();
		trace.pop();
		gx = gRK95(x,gx,dt);
	}
	return gx;
}

#endif
