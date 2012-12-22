#include "Skeleton.h"



/* ** ** ** ** ** DEPRECATED STUFF
void C2S::buildRings(){

	CorrPtr cp = cps.front();

	if (!cp->corr())
		cp = cp->nextCorr();

	RingPtr rpre = 0;

	for( ; cp; cp = cp->nextCorr()){
		if (cp->getFlag(0))
			continue;
		RingPtr rng = new Ring(cp);
		if(rpre)
			rng->setPrevNext(rpre,0);
		else 
			_rings.push_back(rng);
		rpre = rng;
		//visit
		cp->setFlag(0);
		cp->corr()->setFlag(0);

	}
}

void C2S::rebuildRingsByPerspective(){

	double rads[100];
	int ri = 0;
	double radmax =0;
	for(RingPtr r = _ring; r; r = r->next()){
		double rad = r->getNiceRadius();
		rads[ri++] = rad;
		if (radmax < rad)
			radmax = rad;
	}		
	ri =0;
	for(RingPtr r = _ring; r; r = r->next()){
		double k = rads[ri++] / radmax;
		double newr = sqrt(k) * radmax;
		r->setRadius(newr);//newr);
	}

	//now reorient
	for(RingPtr r = _ring; r; r = r->next()){
		Vec3 n = r->next()?r->next()->P():r->P();
		n = r->prev()?(n-r->prev()->P()):(n-r->P());
		r->reorient(n.normalize());
	}
}
/*
void C2S::buildSpine(){

	double rad[100];
	CorrPtr corrs[100];
	CorrPtr cp = cps.front();
	if (!cp->corr())
		cp = cp->nextCorr();
	int id =0;
	double radmax =0;
	for( ; cp && !cp->isSplit(); cp = cp->nextCorr()){
		if (cp->getFlag(0))
			continue;
		double r = ((cp->P()-cp->corr()->P())%(Eye::get()->P*2.0-cp->P()-cp->corr()->P()).normalize()).norm()/2.0;
		rad[id] = r;
		if (r>radmax)
			radmax = r;
		corrs[id] = cp;
		id++;
		//visit
		cp->setFlag(0);
		cp->corr()->setFlag(0);
	}

	for(int i=0; i<id; i++){
		double k = rad[i] / radmax;
		rad[i] = sqrt(k) * radmax;
	}

	Vec3 * pts = new Vec3[id];
	double dmax = 0;
	for(int i=0; i<id; i++){
		Vec3 o = C2S::getOrgbyRad(corrs[i], rad[i]);
		double d = o.norm();
		if (d>dmax)
			dmax = d;
		pts[i] = o;
	}

	//compress in z
	for(int i=0; i<id; i++){
		 Vec3 o = pts[i];
		 double d = o.norm();
		 double a = PZ / (o.normalize()*Eye::get()->N);
		 double t = ((d-a)<0?0:(d-a))*ZCOMPRESS;
		 o = o.normalize()*a;//(a+t);
		 pts[i] = o;
	}

	ArrCurve *ac = new ArrCurve(pts, id,true);
	ac->drawAux(true);
	_zcurve = ac;

}

void C2S::filterSpine(){
	for (list<CurvePtr>::iterator it = _spine.begin(); it != _spine.end(); it++)
		filterSpine((*it));
}
void C2S::filterSpine(CurvePtr sp){
	ArrCurve* ac = ((ArrCurve*)sp);
	ac->movingAverage(5, true);
	ac->resample();
}

list<CurvePtr> C2S::buildSpine(){

	list<CurvePtr> spine;

	double rad[100];
	CorrPtr corrs[100];
	int id =0;
	double radmax =0;

	for(list<CorrPtr>::iterator it = cps.begin(); it!=cps.end(); it++){
		CorrPtr cp = (*it);
		if (!cp->corr())
			cp = cp->nextCorr();
		for( ; cp; cp = cp->nextCorr()){
			if (cp->getFlag(0))
				continue;
			double r = ((cp->P()-cp->corr()->P())%(Eye::get()->P*2.0-cp->P()-cp->corr()->P()).normalize()).norm()/2.0;//nice radious
			rad[id] = r;
			if (r>radmax)
				radmax = r;
			corrs[id] = cp;
			cp->storeD(rad[id]);
			id++;
			//visit
			cp->setFlag(0);
			cp->corr()->setFlag(0);
		}
	}
	//normalize by max radious
	for(int i=0; i<id; i++){
		double k = rad[i]*0.94+radmax*0.06;
		corrs[i]->storeD(k);
	}

	//begin traversal
	list<CorrPtr> que;
	CorrPtr cp = cps.front();
	if (!cp->corr())
		cp = cp->nextCorr();
	que.push_back(cp);
	while(!que.empty()){
		CorrPtr cpend = 0;
		CorrPtr cp0 = que.front();
		que.pop_front();
		ArrCurve* ac = getSpine(cp0, cpend);
		NURBS * nc = NURBS::create(ac);

		spine.push_back(nc);
		cp0->setSpine(nc, cpend);

		if (cpend && cpend->isSplit()){
			for(CorrPtr cpi = cpend->nextCorr(); cpi && cpi!=cpend && cpi!=cpend->corr(); cpi = cpi->corr()->nextCorr())
				que.push_back(cpi);
		}
	}
	return spine;
}
*/
