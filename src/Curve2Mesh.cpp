#include <map>
#include "Curve2Mesh.h"

bool DRAW_AUX = true;

CorrPoint::CorrPoint(const Vec3& p, CurvePtr c){
	_p0 = p;
	_c = c;
	_prev = 0;
	_next = 0;
	_corr = 0;
	_flag  = 0;
	_isshooter = false;
	_spine = 0;
}
void CorrPoint::updateNT(){
	_t0 = ((_next?_next->P():_p0) - (_prev?_prev->P():_p0)).normalize();
	//_n0 = (Eye::get()->N%_t0).normalize();
}

void CorrPoint::drawAll(){

	if (!DRAW_AUX)
		return;
	glColor3f(0.0f,1.0f,0.0f);
	glPointSize(4.0f);
	glBegin(GL_POINTS);
	for(CPPtr cp = this; cp; cp = cp->next()){	
		glVertex3f(cp->P().x, cp->P().y, cp->P().z);
		if (cp->next() == this)
			break;
	}
	glEnd();

	//corr
	glBegin(GL_LINES);
	for(CorrPtr cp = this; cp; cp =(CorrPtr) cp->next()){
		if (cp->_corr){
			
			if (cp->isSplit())
				glColor3f(0.0f,0.0f,1.0f);
			else if (cp->isTip())
				glColor3f(0.0f,0.0f,0.0f);
			else
				glColor3f(0.0f,1.0f,0.0f);

			Vec3 p1 = cp->_corr->_p0;
			glVertex3f(cp->_p0.x, cp->_p0.y, cp->_p0.z);
			glVertex3f(p1.x, p1.y, p1.z);
			/*
			if (DRAW_AUX){
				glColor3f(0.98f,0.98f,0.98f);
				Vec3 pp1 = Eye::get()->P+(cp->_p0 - Eye::get()->P).normalize() * 5.0;
				Vec3 pp2 = Eye::get()->P+(cp->_corr->_p0 - Eye::get()->P).normalize() * 5.0;
				
				glVertex3f(Eye::get()->P.x, Eye::get()->P.y, Eye::get()->P.z);
				glVertex3f(pp1.x, pp1.y, pp1.z);
				
				glVertex3f(Eye::get()->P.x, Eye::get()->P.y, Eye::get()->P.z);
				glVertex3f(pp2.x, pp2.y, pp2.z);
			}*/
		}
		if (cp->next() == this)
			break;

	}
	glEnd();
}

CorrPtr CorrPoint::corr(){return _corr;}

CorrPtr CorrPoint::tipCorr(){
	CorrPtr fp = 0;
	for(CorrPtr cp=this; cp; cp = (CorrPtr)cp->_prev){
		if ( cp->isTip())
			return cp;
		if (cp->_prev == this)
			break;
	}
	return 0;
}

CorrPtr CorrPoint::nextCorr(){
	int count = 0;
	CPPtr l = this->last();
	/*for(CPPtr cp = _next; cp && cp!=this; cp = cp->next()){
		if ( ( (CorrPtr)cp)->_corr)
			return (CorrPtr)cp;
		//cout<<"cp:"<<cp<<endl;
		count++;
	}*/

	return 0;
}

CorrPtr CorrPoint::prevCorr(){
	for(CPPtr cp = _prev; cp && cp!=this; cp = cp->prev())
		if (( (CorrPtr)cp)->_corr)
			return (CorrPtr)cp;
	return 0;
}

CorrPtr CorrPoint::corrNextCorr(){
	if (!_corr)
		return 0;
	return _corr->nextCorr();
}

CorrPtr CorrPoint::corrPrevCorr(){
	if (!_corr)
		return 0;
	return _corr->prevCorr();
}

void CorrPoint::setCorr(CPPtr cp){
	if (!cp)
		return;
	if (_corr){
		_corr->_corr = 0;
		_isshooter = false;
	}
	_isshooter = true;
	_corr = (CorrPtr)cp;
	_corr->_corr = this;
}

void CorrPoint::discardCorr(){
	if (_corr)
		_corr->_corr = 0;
	_corr = 0;
}

bool CorrPoint::isIntersecting(CorrPtr cp){

	CorrPtr nc = nextCorr();
	//CorrPtr pc = prevCorr();
/*
	Vec3 n0 = cp->_p0 - _p0;
	double len = n0.norm();
	n0 = n0.normalize();

	if (nc){
		Vec3 n1 = (nc->_corr->_p0 - nc->_p0).normalize();
		double t = getIntersectionDist( _p0, n0, nc->_p0, n1);
		if (t>0 && t<len+0.001) 
			return true;
	}

	if (pc){
		Vec3 n1 = (pc->_corr->_p0 - pc->_p0).normalize();
		double t = getIntersectionDist( _p0, n0, pc->_p0, n1);
		if (t>0 && t < len+0.001) 
			return true;
	}
*/
	return false;
}

bool CorrPoint::isSplit(){
	if (!_corr)
		return false;

	CorrPtr nc0 = nextCorr();
	CorrPtr pc0 = prevCorr();

	CorrPtr nc1 = _corr->nextCorr();
	CorrPtr pc1 = _corr->prevCorr();

	if (!(nc0&&pc0&&pc1&&nc1))
		return false;

	if (nc0->_corr!=pc1 || nc1->_corr!=pc0)
		return true;

	return false;
}

bool CorrPoint::isTip(){

	if (!_corr)
		return false;

	CorrPtr c0n = nextCorr();
	CorrPtr c0p = prevCorr();

	CorrPtr c1n = _corr->nextCorr();
	CorrPtr c1p = _corr->prevCorr();

	if ( (!c0n&&!c1p) || (!c0p&&!c1n) )
		return true;

	if ( c0n == _corr || c0p == _corr)
		return true;

	return false;
}

void CorrPoint::setSpine(CurvePtr sp, CorrPtr end){
	_spine = sp;
	_sptip = end;
	_spdir = 1;
	if (_sptip){
		_sptip->_spine = sp;
		_sptip->_sptip = this;
		_sptip->_spdir = -1;
	}
}

//+forward, -backward, 0 bothways
int CorrPoint::getFreeCount(int dir){
	if (_corr)
		return 0;
	if (dir==0)
		return getFreeCount(-1)+getFreeCount(1)-1;

	CorrPtr it = this;
	int count = 0;
	while( it && !it->_corr){
		it = (CorrPtr) ((dir>0)?it->_next:it->_prev);
		count++;
	}
	return count;
}

CorrPtr CorrPoint::getShoot(Vec3 p, Vec3 n){
		
	CorrPtr to = this;
	int dir = -1;
	if (this->_next){
		double c0 = abs((to->_p0 - p).normalize()*n);
		double c1 = abs((to->next()->P() - p).normalize()*n);
		if (c1<c0)
			dir =1;
	}

	CorrPtr maxcp = 0; 
	double maxcos = 0;
				
	while(to){
		Vec3 nn = (to->_p0 - p).normalize();
		double cos = abs(nn*n);
		if (cos > maxcos && cos>0.707){
			maxcp = to;
			maxcos = cos;
		}
		if (dir>0)
			to = (CorrPtr)to->_next;
		else
			to = (CorrPtr)to->_prev;
	}
	return maxcp;
}

CorrPtr CorrPoint::sampleCurve(CurvePtr c, double step){

	CorrPtr pre = 0;
	CorrPtr tip = 0;
	int newsize;
	cout<<"curve s:"<<c->size()<<" l:"<<c->length()<<" al:"<<((ArrCurve*)c)->length()<<" as:"<<((ArrCurve*)c)->size()<<" all:"<<ArrCurve::length(c->toArr(), c->size())<<endl;
	Vec3 * p = ArrCurve::resample(c->toArr(), c->size(), step, newsize);
	for(int i = 0; i<newsize; i++){
		CorrPtr cp = new CorrPoint(p[i], c);
		if (pre)
			cp->setPrevNext(pre,0);
		else 
			tip = cp;
		pre = cp;
	}

	for(CorrPtr cp = tip; cp; cp = (CorrPtr)cp->next())
		cp->updateNT();	

	if (c->isClosed()){
		CorrPtr end = (CorrPtr)tip->last();
		tip->_prev = end;
		end->_next = tip;
	}

	return tip;
}

void C2M::addCurve(CurvePtr c){
	CorrPtr cp = CorrPoint::sampleCurve(c, STEP);
	cps.push_back(cp);
}

void C2M::drawGL(){
	for (list<CorrPtr>::iterator it = cps.begin(); it != cps.end(); it++)
		(*it)->drawAll();

	if (_mesh)
		_mesh->drawGL();
}

C2M::C2M(){
	sman = StrokeManager::getManager();
	_mesh = 0;
}

void C2M::resampleCurves(){
	cps.clear();
	for (list<StrokePtr>::iterator it = sman->getStrokes()->begin(); it != sman->getStrokes()->end(); it++){
		addCurve((*it));
		int sz = (*it)->size();
	}
}

void C2M::buildCorrespondances(){
	for(list<CorrPtr>::iterator it =cps.begin(); it!=cps.end(); it++){
		CorrPtr cp0 = (*it);
		int c = 0;
		for(CorrPtr cp = cp0; cp && cp->next()!=cp0; ){
			CPPtr last = cp->last();
			CPPtr ln = last->next();
			findCorr(cp);
			cp->setCorr(0);
			//cout<<cp<<endl;
			for(int i=0; cp && i<4 && cp->next()!=cp0; cp = (CorrPtr)cp->next())
				i++;
			c++;
		}
	}

	//CorrPtr corr = 0;
	//CorrPtr first = cps.front();
		/*if (corr){
			double min = -1;
			corr = findCorr(cp, (CorrPtr)corr->prev(), min);
		}else*/
}


CorrPtr C2M::spot2Shoot(CorrPtr cp){
	//return (CorrPtr)cp->go(4);
	for(int i=0; cp && i<4; cp = (CorrPtr)cp->next())
		i++;
	return cp;
}

CorrPtr C2M::findCorr(CorrPtr cp0){
	double kmin =  99999;
	double min = kmin;
	CorrPtr corr = 0;
	for(list<CorrPtr>::iterator it =cps.begin(); it!=cps.end(); it++){

		//CorrPtr last = (CorrPtr)(*it)->last();
		//cout<<cp0<<"----"<<last<<"*"<<cps.size()<<endl;
		CorrPtr _corr = findCorr(cp0, (CorrPtr)(*it)->last(), min);

		
		if (min<kmin){
			kmin = min;
			corr = _corr;
		}
	}
	return corr;
}

CorrPtr C2M::findCorr(CorrPtr cp0, CorrPtr cp1, double& kmin){

	bool skip = (kmin<0)?false:true;
	kmin = 99999;
	CorrPtr corr = 0;
	int count = 0;
	CPPtr last = cp0->last();
	for(int i=0; i<1;i++){
		//last->setPrevNext(last->prev(),0);
		CPPtr last2 = cp0->last();
	}
	/*
	for(CorrPtr cpi=cp1; cpi && cpi!=cp0; cpi = (CorrPtr)cpi->prev()){
		count++;
		last->setPrevNext(last->prev(),0);
		CPPtr last2 = cp0->last();

	}
	
	/*{
	  //CPPtr last2 = cp0->last();
	/*	if (cp0 == cpi || cp0->corr() || cpi->corr() || cp0->isIntersecting(cpi))
			continue;
	
		//	if (cp0 == cpi || cpi->corr()){ if (skip) continue; else break;}

			
		Vec3 c1mc0 =cpi->P()-cp0->P();
		double cos = fabs(cp0->T()*c1mc0.normalize()); 
		if ( cos > 0.707)
				continue;

		double cosi = fabs(cpi->T()*(-c1mc0.normalize()));
		if ( cosi > 0.707)
				continue;
	
		double r=999999999; 
		//double dist0 = abs(getIntersectionDist(cp0->P(),cp0->T(), cpi->P(), cpi->T() ));
		//double dist1 = abs(getIntersectionDist(cpi->P(),cpi->T(), cp0->P(), cp0->T() ));

		Vec3 pAB = getIntersection(cp0->P(),cp0->T(), cpi->P(), cpi->T());
		double dist0 = (cp0->P() - pAB).norm();
		double dist1 = (cp1->P() - pAB).norm();
		double diff = abs(dist0-dist1);

		if (diff <STEP*2)
			r =c1mc0.norm()*0.5 / cos;
		else
			r = (cp0->P()-cpi->P()).norm()/(2.0 * cos);
		
		double k = r; //+ diff;
		if ( kmin > k){
			kmin = k;
			corr = cpi;
		}
		count++;
	}*/
	return corr;
}

void C2M::filter(int num){

	map<CorrPtr, CPPtr> new_corrs;		
		for(int i=0; i<num; i++){
			new_corrs.clear();
			for(list<CorrPtr>::iterator it =cps.begin(); it!=cps.end(); it++)
				for(CorrPtr cp = (*it); cp && cp!=(*it)->prev(); cp = (CorrPtr)cp->next()){
					
					if (!cp->isShooter() || cp->isSplit() || cp->isTip() )
						continue;
						
					CorrPtr pre  = cp->prevCorr();
					CorrPtr next = cp->nextCorr();
					
					if (!pre || !next )
						continue;
					
					int dir = pre->corr()->find(next->corr());
					if (dir == 0)
							continue;
					
					CPPtr corr = pre->corr()->go(dir/2);
					if (corr)
						new_corrs.insert( pair<CorrPtr, CPPtr>(cp, corr) );				
				}

			for(map<CorrPtr, CPPtr>::iterator it = new_corrs.begin(); it!=new_corrs.end(); it++)
				(*it).first->setCorr((*it).second);
	}
}

void C2M::bumFilter(){
	for(list<CorrPtr>::iterator it = cps.begin(); it!=cps.end(); it++)
		bumFilter(*it);
}

void C2M::bumFilter(CorrPtr _cp0){
	CorrPtr cp0 = _cp0;
	while(cp0){			
		CorrPtr cp0n = cp0->nextCorr();							
		CorrPtr cp0nn = (!cp0n)?0:cp0n->nextCorr();			
		if (!cp0nn)			break;			
		cp0n->discardCorr();

		Vec3 n  = (cp0->P() - cp0nn->corr()->P()).normalize();
		Vec3 nn = (cp0nn->P() - cp0->corr()->P() ).normalize();

		Vec3 cp0_u  =  (cp0->P() - cp0->corr()->P()).normalize();
		Vec3 cp0nn_u = (cp0nn->P() - cp0nn->corr()->P()).normalize();

		Vec2 pAB  = getIntersection(cp0->P(), cp0_u, cp0nn->P(), cp0nn_u);
		Vec2 pMid = getIntersection(cp0->P(), n, cp0nn->P(), nn);

		Vec3 u = (pAB - pMid).normalize();
			
		CorrPtr newcp0 = cp0->getShoot(pMid, u);
		CorrPtr newcp1 = cp0->corr()->getShoot(pMid, u);
			
		if (newcp0)	
			newcp0->setCorr(newcp1);
		cp0 = newcp0;			
	}
}

void C2M::growCaps(){

	for(list<CorrPtr>::iterator it = cps.begin(); it!=cps.end(); it++){
		CorrPtr cp0 = (*it); 
		for(CorrPtr cp = cp0; cp; cp = (CorrPtr)cp->next() ){
			
			if (cp->corr() && (cp->nextCorr() == cp->corr() || (!cp->nextCorr() && !cp->corr()->prevCorr())) ){
				
				CorrPtr cpc = cp->corr();
				int n = cp->find(cpc);
				while(n>8){
					cp = (CorrPtr)cp->go(4);
					cpc = (CorrPtr)cpc->go(-4);
					cp->setCorr(cpc);
					n-=8;
				}
			}
			if (cp->next() == cp0)
				break;
		}
	}
}

int findClosest(Vec3 p, Vec3 * ps, int sz, int start, int inc){
	double min_dist = 99999999;
	int min = -1; 
	for(int i = start+inc; i%sz != start; i+=inc){
		double d = (p-ps[i%sz]).norm();
		if (d < min_dist){
			min_dist = d;
			min = i%sz;
		}
	}
	return min;
}

void  C2M::buildMesh(int seg){

	_mesh = new Mesh();
	VertexPtr * vs0 = 0; 
	Vec3 * ps = 0;
	int vc  = seg*10;

	for (list<StrokePtr>::iterator it = sman->getStrokes()->begin(); it != sman->getStrokes()->end(); it++){
		ps = (*it)->curve()->toArr(vc);
		VertexPtr * vs = 0;
		if (!vs0){
			vs0 = new VertexPtr[seg];
			for(int i = 0; i<seg; i++){
					vs0[i] = new Vertex();
					vs0[i]->setP(ps[i*10]);
			}
		}else{
			vs = new VertexPtr[seg*10];
			int vid = 0;
			for(int i = 0; i<seg; i++){
				vs[i] = new Vertex();
				int ind = i;
				vid = findClosest(vs0[i]->getP(), ps, vc, vid, 1);
				vs[i]->setP(ps[vid]);
			}
		}

		if (vs0 && vs){
			for(int j=0; j<seg-1; j++){
				_mesh->addQuad(vs[j], vs[j+1], vs0[j+1], vs0[j]);
			}
			vs0 = vs;
		}
	}

}