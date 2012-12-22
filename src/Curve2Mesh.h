#ifndef _H_C2M
#define _H_C2M

#include <map>
#include "Stroke.h"
#include "Mesh.h"


class CorrPoint;
class Ring;
#define CorrPtr CorrPoint*
#define RingPtr Ring*
#define STEP 0.005

static const double ZCOMPRESS = 0.25;
extern bool DRAW_AUX;
class CorrPoint:public CurvePoint{

	//corr
	CorrPtr _corr;
	bool _isshooter;
	unsigned int _flag;

	void updateNT();
	double _storeD;

	//spinal fields
	CurvePtr _spine;
	CorrPtr  _sptip;
	int		 _spdir;

public:

	CorrPoint(const Vec3& p, CurvePtr c);
	void drawAll();
	
	CorrPtr corr();
	CorrPtr tipCorr();
	CorrPtr nextCorr();
	CorrPtr prevCorr();
	CorrPtr corrNextCorr();
	CorrPtr corrPrevCorr();

	void setCorr(CPPtr cp);
	void discardCorr();

	bool isTip();
	bool isSplit();
	bool isIntersecting(CorrPtr cp);
	int  getFreeCount(int dir = 1);
	bool isShooter(){return _isshooter;};

	CorrPtr getShoot(Vec3 p, Vec3 n);
	void setSpine(CurvePtr sp, CorrPtr end);
	CurvePtr spine(){return _spine;};
	int spinalDir(){return _spdir;};
	CorrPtr tip(){return _sptip;};

	void storeD(double d){_storeD = d;};
	double retrieveD(){return _storeD;};

	void setFlag(unsigned int b){_flag |= (1 << b);};
	bool getFlag(unsigned int b){return _flag & (1 << b);};

	static CorrPtr sampleCurve(CurvePtr c, double step);
	static CorrPtr sampleCurve(Vec3 * p, int size, double step);
};

class C2M{

	list<CorrPtr> cps;
	StrokeManager* sman;

	CorrPtr spot2Shoot(CorrPtr cp);
	CorrPtr findCorr(CorrPtr);
	CorrPtr findCorr(CorrPtr, CorrPtr,  double&);
	ArrCurve* getSpine(CorrPtr cp, CorrPtr& endcp);

	void bumFilter(CorrPtr);
	void addCurve(CurvePtr c);
	void growCap(CorrPtr);

public:

	C2M();	
	void resampleCurves();
	void drawGL();

	void buildCorrespondances();
	void buildCorrespondance();
	void filter(int num);
	
	void bumFilter();
	//void filterSpine();
	//list<CurvePtr> buildSpine();
	
	list<CorrPtr> getCorrs(){return cps;};
	void growCaps();

	void buildMesh(int);

	MeshPtr _mesh;
};


class CurveBoss{

	C2M * _c2m;
	
	StrokeManager * _sm;

	static CurveBoss * _boss;
	int _step;

public:

	CurveBoss();
	void buildSpine();
	void buildMesh();

	void reset();
	void drawGL();

	void saveMesh(char *fname);

	static CurveBoss* getBoss();
	void step();

};

#endif