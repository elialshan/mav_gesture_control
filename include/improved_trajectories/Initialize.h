#ifndef INITIALIZE_H_
#define INITIALIZE_H_

#include "DenseTrackStab.h"


void InitTrackInfo(TrackInfo* trackInfo, int g_track_length, int g_init_gap)
{
	trackInfo->length = g_track_length;
	trackInfo->gap = g_init_gap;
}

void InitDescMat(int height, int width, int nBins, DescMat* descMat)
{
	descMat->height = height;
	descMat->width = width;
	descMat->nBins = nBins;

	long size = height*width*nBins;
    descMat->size = size;
	descMat->desc = (float*)malloc(size*sizeof(float));
	memset(descMat->desc, 0, size*sizeof(float));	
}

void ReleDescMat(DescMat* descMat)
{
	free(descMat->desc);
}

void InitDescInfo(DescInfo* descInfo, int nBins, bool isHof, int size, int g_nxy_cell, int g_nt_cell)
{
	descInfo->nBins = nBins;
	descInfo->isHof = isHof;
	descInfo->nxCells = g_nxy_cell;
	descInfo->nyCells = g_nxy_cell;
	descInfo->ntCells = g_nt_cell;
	descInfo->dim = nBins*g_nxy_cell*g_nxy_cell;
	descInfo->height = size;
	descInfo->width = size;
}


#endif /*INITIALIZE_H_*/
