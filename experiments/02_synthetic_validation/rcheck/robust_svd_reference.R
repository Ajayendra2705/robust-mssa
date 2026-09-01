#!/usr/bin/env Rscript
# Authoritative cross-check: run the two robust SVD algorithms of Rodrigues et al.
# (2020, Entropy) using the ORIGINAL reference implementations,
#   * RLSSA -> pcaMethods::robustSvd   (L1-norm; Hawkins, Liu & Young 2001)
#   * RHSSA -> RobRSVD::RobRSVD        (Huber;   Zhang, Shen & Huang 2013)
# on the same fixture matrix H the Python side used, and write their leading left
# singular vectors so run_rcheck.py can measure subspace agreement.
#
# Run (from this directory):
#   Rscript robust_svd_reference.R . 2
#
# Install (into a user library if the main one is not writable):
#   install.packages("https://cran.r-project.org/src/contrib/Archive/RobRSVD/RobRSVD_1.0.tar.gz",
#                    repos = NULL, type = "source")   # archived on CRAN
#   BiocManager::install("pcaMethods"); install.packages("matrixStats")
#
# ---------------------------------------------------------------------------
# Two corrections against the first draft of this script, both found by actually
# running it. They matter for what is being validated:
#
#   1. The pcaMethods function is `robustSvd`, not `robustSVD`.
#   2. `RobRSVD` has NO `rough` argument. Robustness is controlled by `irobust`,
#      which DEFAULTS TO FALSE — so the call in the first draft would not have
#      been the Huber robust SVD at all, but the plain regularized SVD. The Huber
#      variant of the paper is
#          RobRSVD(M, irobust = TRUE, huberk = 1.345, uspar = 0, vspar = 0)
#      where huberk = 1.345 is the paper's tuning constant and uspar = vspar = 0
#      switches off the roughness penalties, leaving the plain Huber robust SVD.
#
# RobRSVD returns a single rank-1 layer as list(s, u, v, diagout) — the singular
# value is `s` (there is no `d`) and u, v are unit-norm — so rank r is reached by
# deflating r times.
# ---------------------------------------------------------------------------

args  <- commandArgs(trailingOnly = TRUE)
indir <- if (length(args) >= 1) args[1] else "."
r     <- if (length(args) >= 2) as.integer(args[2]) else 2L

H <- as.matrix(read.csv(file.path(indir, "H.csv"), header = FALSE))
storage.mode(H) <- "double"

## The two references are run INDEPENDENTLY under try(). pcaMethods::robustSvd
## fails on wide matrices, and if that aborted the script the Huber comparison --
## the primary algorithm -- would be lost on exactly the fixtures that matter most.
## Each writes its own file; a missing file means "this reference could not run".

## --- RLSSA: L1-norm robust SVD (pcaMethods::robustSvd) ----------------------
suppressMessages(library(pcaMethods))
rl <- try(robustSvd(H), silent = TRUE)
if (inherits(rl, "try-error")) {
  cat("robustSvd FAILED:", conditionMessage(attr(rl, "condition")), "\n")
} else {
  write.table(rl$u[, seq_len(r), drop = FALSE], file.path(indir, "R_U_l1.csv"),
              sep = ",", row.names = FALSE, col.names = FALSE)
  cat("wrote R_U_l1.csv\n")
}

## --- RHSSA: Huber robust SVD (RobRSVD), huberk = 1.345 ----------------------
suppressMessages(library(RobRSVD))
hub <- try({
  M     <- H
  U_hub <- matrix(0.0, nrow = nrow(H), ncol = r)
  for (i in seq_len(r)) {
    fit <- RobRSVD(M, irobust = TRUE, huberk = 1.345, uspar = 0, vspar = 0)
    u <- as.numeric(fit$u); v <- as.numeric(fit$v); s <- as.numeric(fit$s)
    U_hub[, i] <- u
    M <- M - s * (u %*% t(v))          # robust deflation
  }
  U_hub
}, silent = TRUE)
if (inherits(hub, "try-error")) {
  cat("RobRSVD FAILED:", conditionMessage(attr(hub, "condition")), "\n")
} else {
  write.table(hub, file.path(indir, "R_U_huber.csv"),
              sep = ",", row.names = FALSE, col.names = FALSE)
  cat("wrote R_U_huber.csv\n")
}

cat("done (r =", r, ")\n")
