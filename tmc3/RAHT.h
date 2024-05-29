/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2018, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once
#include <cstdint>

#include "FixedPoint.h"
#include "quantization.h"
#include "hls.h"
#include "PCCTMC3Common.h"
#include <vector>
#include <queue>

namespace pcc {

class ModeCoder
{
protected:
  AdaptiveBitModel ctxInterPred;
  AdaptiveBitModel ctxIntraPred;
public:
  void reset()
  {
    ctxInterPred.probability = 0x8000;
    ctxIntraPred.probability = 0x8000;
  }
  void updateInterIntraEnabled(bool flag, bool flag2, int flag3) {

  }
  int decodeMode(const bool& enableInter, const bool& enableIntra) { throw std::runtime_error("not implemented"); }
  void _encodeMode(int predMode, const bool& enableInter, const bool& enableIntra,const bool& code = true) {
    throw std::runtime_error("not implemented");
  }
};

class ModeEncoder: public ModeCoder 
{
  typedef decltype(AdaptiveBitModel::probability) probaType;
  EntropyEncoder* arith;
  std::deque<int> modeBuffer;
  std::deque<bool> curLayerEnableInter;
  std::deque<bool> curLayerEnableIntra;
public:
  void set(EntropyEncoder* coder) {
    arith = coder;
    modeBuffer.resize(0);
    curLayerEnableInter.resize(0);
    curLayerEnableIntra.resize(0);
  }
  ModeEncoder()
    : ModeCoder()
    , arith(nullptr)
  {
    
  }
  ~ModeEncoder() { if (arith) flush(); }
  void flush()
  {
    int rahtFilterLayerIdx = 0;
    for (int depth = 0; depth < modeBuffer.size(); ++depth) {
      _encodeMode(modeBuffer[depth], curLayerEnableInter[depth], curLayerEnableIntra[depth], true);
    }
    modeBuffer.clear();
    curLayerEnableInter.clear();
    curLayerEnableIntra.clear();
  }
  
  void _encodeMode(int predMode,const bool& enableInter,const bool& enableIntra,const bool& writeOut = false)
  {
    if (!writeOut) {
      modeBuffer.push_back(predMode);
      curLayerEnableInter.push_back(enableInter);
      curLayerEnableIntra.push_back(enableIntra);
    }
    else {
      if (enableInter) {
        const bool& isInter = predMode == 1;
        arith->encode(isInter, ctxInterPred);
        if (enableIntra && !isInter) {
          const bool& isIntraPred = predMode == 0;
          arith->encode(isIntraPred, ctxIntraPred);
        }
      }
      else if (enableIntra) {
        const bool& isIntraPredMode = predMode == 0;
        arith->encode(isIntraPredMode, ctxIntraPred);
      }
    }
  }
};


class ModeDecoder : public ModeCoder {
  EntropyDecoder* arith;

public:
  ModeDecoder() : ModeCoder(), arith(nullptr) {}

  void set(EntropyDecoder* coder) { arith = coder; }

  int decodeMode(const bool& enableInter, const bool& enableIntra)
  {
    if (enableInter) {
      bool isInter;
      isInter = arith->decode(ctxInterPred);
      if (isInter)
        return 1;
      if (!enableIntra)
        return 0;
      bool isIntraPred = arith->decode(ctxIntraPred);
      if (isIntraPred)
        return 0;
      return 2;
    }
    else if (enableIntra) {
      bool isIntraPred = arith->decode(ctxIntraPred);
      if (isIntraPred)
        return 0;
      return 1;
    } 
	else
      return -1;
  }
};

void regionAdaptiveHierarchicalTransform(
  const RahtPredictionParams& rahtPredParams,
  const QpSet& qpset,
  const Qps* pointQPOffset,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool removeRoundingOps,
  AttributeInterPredParams& attrInterPredParam,
  ModeEncoder& encoder);

void regionAdaptiveHierarchicalInverseTransform(
  const RahtPredictionParams &rahtPredParams,
  const QpSet& qpset,
  const Qps* pointQpOffset,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool removeRoundingOps,
  AttributeInterPredParams& attrInterPredParams,
  ModeDecoder& decoder);

struct PCCRAHTACCoefficientEntropyEstimate {
  PCCRAHTACCoefficientEntropyEstimate()
  { init(); }

  PCCRAHTACCoefficientEntropyEstimate(
    const PCCRAHTACCoefficientEntropyEstimate &other) = default;

  PCCRAHTACCoefficientEntropyEstimate&
  operator=(const PCCRAHTACCoefficientEntropyEstimate&) = default;

  void resStatUpdate(int32_t values, int k);
  void init();
  void updateCostBits(int32_t values, int k);
  double costBits() { return sumCostBits; }
  void resetCostBits() { sumCostBits = 0.; }

private:
  // Encoder side residual cost calculation
  static constexpr unsigned scaleRes = 1 << 20;
  static constexpr unsigned windowLog2 = 6;
  int probResGt0[3];  //prob of residuals larger than 0: 1 for each component
  int probResGt1[3];  //prob of residuals larger than 1: 1 for each component
  double sumCostBits;
};



struct PCCRAHTComputeLCP {
  int8_t computeLastComponentPredictionCoeff(
    const bool& enableNonPred,
    const bool& enableIntraPred,
    int m,
    int64_t coeffs[][3], 
	int64_t nonPredCoeffs[][3], 
	int8_t& nonPredLcpCoeff,
    int64_t intraPredCoeffs[][3],
    int8_t& intraPredLcpCoeff);

private:
  int64_t sumk1k2 = 0;
  int64_t sumk1k1 = 0;
  int64_t nonPredSumk1k2 = 0;
  int64_t nonPredSumk1k1 = 0;
  int64_t intraPredSumk1k2 = 0;
  int64_t intraPredSumk1k1 = 0;
  std::queue<int64_t> window1;
  std::queue<int64_t> window2;
  std::queue<int64_t> nonPredWindow1;
  std::queue<int64_t> nonPredWindow2;
  std::queue<int64_t> intraPredWindow1;
  std::queue<int64_t> intraPredWindow2;
};
} /* namespace pcc */
