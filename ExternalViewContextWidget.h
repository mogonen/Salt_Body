/***************************************************************************\

  NAME:         ExternalViewContextWidget.h

  DESCRIPTION:  This sample shows how to instantiate and animate some simple
                mesh objects which are attached to each other

  NOTE:         This file is part of the HueSpace visualization API.
                Copyright (C) 2001-2011 Hue AS. All rights reserved.

\***************************************************************************/

#ifndef EXTERNALVIEWCONTEXT_H_INCLUDED
#define EXTERNALVIEWCONTEXT_H_INCLUDED

// INCLUDES /////////////////////////////////////////////////////////////////

#include <QtCore/qdatetime.h>
#include "HueQtDefaultViewerWidget.h"
#include "ShapeInstanceViewContext.h"
#include "ExternalViewContext.h"
#include "src/Stroke.h"
#include "src/Curve2Mesh.h"
#include "DirectShape.h"

// CLASSES //////////////////////////////////////////////////////////////////

// ExternalViewContextWidget

class ExternalViewContextWidget : public HueQtDefaultViewerWidget
{
  Q_OBJECT;
public:
  ExternalViewContextWidget(QWidget *pParent, int argc, char ** argv);
  ~ExternalViewContextWidget();
  static Hue::ProxyLib::Viewer * pViewer;
  static StrokeManager * strokeman;
  static CurveBoss * cboss;

  virtual void mouseMoveEvent(QMouseEvent *pMouseEvent);
  virtual void mousePressEvent(QMouseEvent *);
  virtual void mouseReleaseEvent(QMouseEvent *);
  virtual void keyReleaseEvent(QKeyEvent * pQKeyEvent);

private:
  Hue::ProxyLib::ExternalViewContext *_pExternalViewContext;
  bool sketching_on;

  Vec3 p0, n0;
  Hue::ProxyLib::DirectShape * _pLineSet;
  void createLineSet();

};

#endif // !defined EXTERNALVIEWCONTEXT_H_INCLUDED
