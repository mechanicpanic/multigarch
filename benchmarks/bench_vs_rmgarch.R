# DCC(1,1) with GARCH(1,1)-normal margins in rmgarch, for comparison with
# multigarch (see bench_vs_rmgarch.py).
#
#   Rscript benchmarks/bench_vs_rmgarch.R panel.csv 5,10,30,50,100 > rmgarch.txt
#
# panel.csv: (T, n) returns, no header. Needs rugarch and rmgarch.
suppressPackageStartupMessages({library(rugarch); library(rmgarch)})
args <- commandArgs(trailingOnly = TRUE)
X <- as.matrix(read.csv(args[1], header = FALSE))
sizes <- as.integer(strsplit(args[2], ",")[[1]])
uspec <- ugarchspec(mean.model = list(armaOrder = c(0, 0), include.mean = FALSE),
                    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                    distribution.model = "norm")
for (n in sizes) {
  spec <- dccspec(uspec = multispec(replicate(n, uspec)), dccOrder = c(1, 1),
                  distribution = "mvnorm")
  t <- system.time(fit <- dccfit(spec, data = X[, 1:n],
                                 fit.control = list(eval.se = FALSE)))["elapsed"]
  cf <- coef(fit)
  cat(sprintf("RESULT n=%d time=%.3f a=%.6f b=%.6f\n", n, t,
              cf["[Joint]dcca1"], cf["[Joint]dccb1"]))
}
