#include "Curve2Mesh.h"

CurveBoss* CurveBoss:: _boss;
CurveBoss::CurveBoss(){
	_step = 0;
	_sm = StrokeManager::getManager();
	_c2m = new C2M();
}

CurveBoss* CurveBoss::getBoss(){
	if (!_boss)
		_boss =  new CurveBoss();
	return _boss;
}

void CurveBoss::drawGL(){
	if (_c2m)
		_c2m->drawGL();

}

void CurveBoss::reset(){
	delete _c2m;
	_c2m = new C2M();
	_step = 0;
}

void CurveBoss::step(){
	_c2m->buildMesh(64);
	/*if (_step == 0 ){
		_c2m->resampleCurves();
		
	}
	if (_step == 1)
		_c2m->buildCorrespondances();

	if (_step == 2)
		_c2m->filter(3);
*/
	_step++;
}

void CurveBoss::buildMesh(){
	_c2m->buildMesh(64);
}

void CurveBoss::buildSpine(){
}

void CurveBoss::saveMesh(char * fname){
	//_c2m->saveMesh(fname);
}
/*
void CurveBoss::reproject(Camera * cam){
	_sm->setCurves(_c2m->getProjectionCurves(cam));
	//_c2m->buildRings();
}*/