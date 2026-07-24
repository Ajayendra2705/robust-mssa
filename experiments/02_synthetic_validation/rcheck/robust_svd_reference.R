#!/usr/bin/env Rscript
# Day-14 authoritative cross-check: run the two robust SVD algorithms of
# Rodrigues et al. (2020, Entropy) using the ORIGINAL reference implementations,
#   * RLSSA -> pcaMethods::robustSVD   (L1-norm; Hawkins, Liu & Young 2001)
#   * RHSSA -> RobRSVD::RobRSVD         (Huber; Zhang, Shen & Huang 2013)
# on the same fixture matrix H that the Python side used, and write their leading
# left singular vectors so compare.py can measure subspace agreement.
#
# Run (from this directory):
#   Rscript robust_svd_reference.R
# Requires: install.packages("RobRSVD"); BiocManager::install("pcaMethods")

args <- commandArgs(trailingOnly = TRUE)
indir  <- if (length(args) >= 1) args[1] else "."
r      <- if (length(args) >= 2) as.integer(args[2]) else 2L

H <- as.matrix(read.csv(file.path(indir, "H.csv"), header = FALSE))
storage.mode(H) <- "double"

## --- RLSSA: L1-norm robust SVD (pcaMethods::robustSVD) ----------------------
suppressMessages(library(pcaMethods))
rl <- robustSVD(H)
U_l1 <- rl$u[, seq_len(r), drop = FALSE]
write.table(U_l1, file.path(indir, "R_U_l1.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)

## --- RHSSA: Huber robust regularized SVD (RobRSVD), d = 1.345 ---------------
## rough = TRUE with uspar = vspar = 0 -> plain Huber robust SVD (no smoothing),
## matching the paper's call. RobRSVD returns one rank-1 layer; deflate for r.
suppressMessages(library(RobRSVD))
M <- H
U_hub <- matrix(0.0, nrow = nrow(H), ncol = r)
for (i in seq_len(r)) {
  fit <- RobRSVD(M, rough = TRUE, uspar = 0, vspar = 0)
  u <- fit$u; v <- fit$v; d <- fit$d
  U_hub[, i] <- u
  M <- M - d * (u %*% t(v))            # robust deflation
}
write.table(U_hub, file.path(indir, "R_U_huber.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)

cat("wrote R_U_l1.csv and R_U_huber.csv (r =", r, ")\n")
