import Mathlib.Data.Real.Basic
import Mathlib.Tactic

theorem endpoint_barrier
    {u τ μ ε t : ℝ}
    (hμ : 0 < μ)
    (hu : u ≤ τ - μ * t + ε)
    (ht : ε / μ < t) :
    u < τ := by
  have hε : ε < μ * t := (div_lt_iff₀ hμ).mp ht
  linarith

theorem ridge_displacement
    {κ ε d : ℝ}
    (hκ : 0 < κ)
    (h : κ * |d| ≤ 2 * ε) :
    |d| ≤ 2 * ε / κ := by
  apply (le_div_iff₀ hκ).2
  simpa [mul_comm] using h

theorem positive_wang_zahl_mass
    {a lowerBound : ℝ}
    (ha : 0 < a)
    (hLower : 0 < lowerBound) :
    0 < a / 2 * lowerBound := by
  positivity

example : (23743 : Nat) ≥ 16010 := by decide
example : (2025 : Nat) < 50125 := by decide
example : (1875 : Nat) < 8625 := by decide
example : (915 : Nat) < 1338 := by decide
example : (12244 : Nat) < 16502 := by decide
example : (693 : Nat) > 414 := by decide
