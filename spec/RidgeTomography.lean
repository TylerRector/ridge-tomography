structure Benchmark where
  psnrMilliDb : Nat
  ssimMilli : Nat
  pooledRmseMicro : Nat
  falseSkeletonCentiPx : Nat
  endpointOverrunMilliPx : Nat
  curveLocalizationMilliPx : Nat
  tangentMilliDeg : Nat
  bendCorrelationMilli : Nat
deriving Repr

def targetPsnrMilliDb : Nat := 16010

def weightedFbp : Benchmark where
  psnrMilliDb := 17773
  ssimMilli := 439
  pooledRmseMicro := 131802
  falseSkeletonCentiPx := 50125
  endpointOverrunMilliPx := 8625
  curveLocalizationMilliPx := 1338
  tangentMilliDeg := 16502
  bendCorrelationMilli := 414

def iterative : Benchmark where
  psnrMilliDb := 23743
  ssimMilli := 710
  pooledRmseMicro := 67075
  falseSkeletonCentiPx := 2025
  endpointOverrunMilliPx := 1875
  curveLocalizationMilliPx := 915
  tangentMilliDeg := 12244
  bendCorrelationMilli := 693

def passesPsnr (result : Benchmark) : Bool :=
  result.psnrMilliDb ≥ targetPsnrMilliDb

def improvesStructure (candidate baseline : Benchmark) : Bool :=
  candidate.falseSkeletonCentiPx < baseline.falseSkeletonCentiPx &&
  candidate.endpointOverrunMilliPx < baseline.endpointOverrunMilliPx &&
  candidate.curveLocalizationMilliPx < baseline.curveLocalizationMilliPx &&
  candidate.tangentMilliDeg < baseline.tangentMilliDeg &&
  candidate.bendCorrelationMilli > baseline.bendCorrelationMilli

def main : IO Unit := do
  IO.println s!"weighted_psnr_millidb={weightedFbp.psnrMilliDb}"
  IO.println s!"iterative_psnr_millidb={iterative.psnrMilliDb}"
  IO.println s!"target_psnr_millidb={targetPsnrMilliDb}"
  IO.println s!"iterative_passes={passesPsnr iterative}"
  IO.println s!"iterative_improves_structure={improvesStructure iterative weightedFbp}"
