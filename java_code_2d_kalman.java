import java.util.*;

class Main {

    static class Vec2 {
        double a, b;

        Vec2(double a, double b) {
            this.a = a;
            this.b = b;
        }
    }

    static class Mat2 {
        double a00, a01, a10, a11;

        Mat2(double a00, double a01, double a10, double a11) {
            this.a00 = a00;
            this.a01 = a01;
            this.a10 = a10;
            this.a11 = a11;
        }

        static Mat2 identity(double x) {
            return new Mat2(x, 0.0, 0.0, x);
        }

        Mat2 add(Mat2 o) {
            return new Mat2(a00 + o.a00, a01 + o.a01, a10 + o.a10, a11 + o.a11);
        }

        Mat2 subtract(Mat2 o) {
            return new Mat2(a00 - o.a00, a01 - o.a01, a10 - o.a10, a11 - o.a11);
        }

        Mat2 multiply(Mat2 o) {
            return new Mat2(
                    a00 * o.a00 + a01 * o.a10,
                    a00 * o.a01 + a01 * o.a11,
                    a10 * o.a00 + a11 * o.a10,
                    a10 * o.a01 + a11 * o.a11
            );
        }
    }

    static class KalmanResult {
        double alpha, beta, predictedY, spread, innovationVariance, gainAlpha, gainBeta;

        KalmanResult(
                double alpha,
                double beta,
                double predictedY,
                double spread,
                double innovationVariance,
                double gainAlpha,
                double gainBeta
        ) {
            this.alpha = alpha;
            this.beta = beta;
            this.predictedY = predictedY;
            this.spread = spread;
            this.innovationVariance = innovationVariance;
            this.gainAlpha = gainAlpha;
            this.gainBeta = gainBeta;
        }
    }

    static class DynamicRegressionKalman {
        Vec2 state;
        Mat2 covariance;
        Mat2 processNoise;
        double measurementNoise;

        DynamicRegressionKalman(
                double alpha0,
                double beta0,
                double uncertainty,
                double alphaNoise,
                double betaNoise,
                double measurementNoise
        ) {
            this.state = new Vec2(alpha0, beta0);
            this.covariance = Mat2.identity(uncertainty);
            this.processNoise = new Mat2(alphaNoise, 0.0, 0.0, betaNoise);
            this.measurementNoise = measurementNoise;
        }

        KalmanResult update(double x, double y) {
            Vec2 predictedState = state;
            Mat2 predictedCov = covariance.add(processNoise);

            double h0 = 1.0;
            double h1 = x;

            double predictedY = h0 * predictedState.a + h1 * predictedState.b;
            double spread = y - predictedY;

            double ph0 = predictedCov.a00 * h0 + predictedCov.a01 * h1;
            double ph1 = predictedCov.a10 * h0 + predictedCov.a11 * h1;

            double s = h0 * ph0 + h1 * ph1 + measurementNoise;

            double k0 = ph0 / s;
            double k1 = ph1 / s;

            state = new Vec2(
                    predictedState.a + k0 * spread,
                    predictedState.b + k1 * spread
            );

            Mat2 kh = new Mat2(k0 * h0, k0 * h1, k1 * h0, k1 * h1);
            covariance = Mat2.identity(1.0).subtract(kh).multiply(predictedCov);

            return new KalmanResult(state.a, state.b, predictedY, spread, s, k0, k1);
        }
    }

    static class RollingStats {
        int window;
        Deque<Double> q = new ArrayDeque<>();
        double sum = 0.0;
        double sumSq = 0.0;

        RollingStats(int window) {
            this.window = window;
        }

        void add(double x) {
            q.addLast(x);
            sum += x;
            sumSq += x * x;

            if (q.size() > window) {
                double old = q.removeFirst();
                sum -= old;
                sumSq -= old * old;
            }
        }

        double mean() {
            return q.isEmpty() ? 0.0 : sum / q.size();
        }

        double std() {
            if (q.size() < 2) return 1.0;
            double m = mean();
            double v = sumSq / q.size() - m * m;
            return Math.sqrt(Math.max(v, 1e-9));
        }

        double z(double x) {
            return (x - mean()) / std();
        }

        int size() {
            return q.size();
        }
    }

    static class MarketPoint {
        double x, y, trueAlpha, trueBeta;

        MarketPoint(double x, double y, double trueAlpha, double trueBeta) {
            this.x = x;
            this.y = y;
            this.trueAlpha = trueAlpha;
            this.trueBeta = trueBeta;
        }
    }

    static class Portfolio {
        int position = 0;
        double prevX = 0.0;
        double prevY = 0.0;
        double prevBeta = 1.0;
        double pnl = 0.0;
        double peak = 0.0;
        double maxDrawdown = 0.0;
        ArrayList<Double> returns = new ArrayList<>();

        double update(double x, double y, double beta, int newPosition, double cost) {
            double stepPnl = 0.0;

            if (prevX != 0.0 || prevY != 0.0) {
                stepPnl = position * ((y - prevY) - prevBeta * (x - prevX));
                if (newPosition != position) {
                    stepPnl -= cost * Math.abs(newPosition - position);
                }
                pnl += stepPnl;
                peak = Math.max(peak, pnl);
                maxDrawdown = Math.max(maxDrawdown, peak - pnl);
                returns.add(stepPnl);
            }

            position = newPosition;
            prevX = x;
            prevY = y;
            prevBeta = beta;

            return stepPnl;
        }

        double meanReturn() {
            if (returns.isEmpty()) return 0.0;
            double s = 0.0;
            for (double r : returns) s += r;
            return s / returns.size();
        }

        double stdReturn() {
            if (returns.size() < 2) return 1.0;
            double m = meanReturn();
            double s = 0.0;
            for (double r : returns) s += (r - m) * (r - m);
            return Math.sqrt(s / (returns.size() - 1));
        }

        double sharpe() {
            return stdReturn() < 1e-9 ? 0.0 : Math.sqrt(252.0) * meanReturn() / stdReturn();
        }
    }

    static MarketPoint generate(int t, Random r) {
        double commonTrend = 100.0 + 0.10 * t + 4.0 * Math.sin(t / 18.0);
        double x = commonTrend + 2.0 * Math.sin(t / 7.0) + r.nextGaussian() * 1.10;

        double regimeShift = t > 90 ? 0.22 : 0.0;
        double trueAlpha = 3.0 + 0.9 * Math.sin(t / 45.0) + (t > 130 ? -0.8 : 0.0);
        double trueBeta = 1.05 + regimeShift + 0.16 * Math.sin(t / 32.0) + 0.06 * Math.cos(t / 13.0);

        double temporaryShock = (t == 55 || t == 118 || t == 160) ? 7.0 * (r.nextBoolean() ? 1 : -1) : 0.0;
        double y = trueAlpha + trueBeta * x + temporaryShock + r.nextGaussian() * 1.35;

        return new MarketPoint(x, y, trueAlpha, trueBeta);
    }

    static int signalToPosition(double z, int currentPosition) {
        if (z > 2.0) return -1;
        if (z < -2.0) return 1;
        if (Math.abs(z) < 0.45) return 0;
        return currentPosition;
    }

    static String posName(int p) {
        if (p == 1) return "LONG_Y";
        if (p == -1) return "SHORT_Y";
        return "FLAT";
    }

    static String row(
            int t,
            double x,
            double y,
            double tb,
            double eb,
            double ta,
            double ea,
            double spread,
            double z,
            int pos,
            double stepPnl,
            double totalPnl
    ) {
        return String.format(
                "%4d | %9.3f | %9.3f | %8.4f | %8.4f | %8.4f | %8.4f | %9.4f | %7.3f | %-7s | %9.4f | %10.4f",
                t, x, y, tb, eb, ta, ea, spread, z, posName(pos), stepPnl, totalPnl
        );
    }

    public static void main(String[] args) {
        Random random = new Random(11);

        DynamicRegressionKalman model = new DynamicRegressionKalman(
                0.0,
                1.0,
                250.0,
                0.003,
                0.00004,
                3.2
        );

        RollingStats spreadStats = new RollingStats(30);
        Portfolio portfolio = new Portfolio();

        double betaAbsError = 0.0;
        double alphaAbsError = 0.0;
        double spreadAbs = 0.0;
        int trades = 0;
        int oldPosition = 0;

        System.out.println("Dynamic Pairs Trading Model using 2D Kalman Filter");
        System.out.println("========================================================================================================================================");
        System.out.println("   t |         X |         Y | TrueBeta |  EstBeta | TrueAlph |  EstAlph |    Spread |  ZScore | Pos     |   StepPnL |   TotalPnL");
        System.out.println("========================================================================================================================================");

        for (int t = 1; t <= 180; t++) {
            MarketPoint p = generate(t, random);
            KalmanResult k = model.update(p.x, p.y);

            spreadStats.add(k.spread);

            double z = spreadStats.size() < 20 ? 0.0 : spreadStats.z(k.spread);
            int newPosition = signalToPosition(z, portfolio.position);

            if (newPosition != oldPosition) trades++;
            oldPosition = newPosition;

            double stepPnl = portfolio.update(p.x, p.y, k.beta, newPosition, 0.08);

            betaAbsError += Math.abs(p.trueBeta - k.beta);
            alphaAbsError += Math.abs(p.trueAlpha - k.alpha);
            spreadAbs += Math.abs(k.spread);

            if (t <= 18 || t % 8 == 0 || Math.abs(z) > 2.0) {
                System.out.println(row(
                        t,
                        p.x,
                        p.y,
                        p.trueBeta,
                        k.beta,
                        p.trueAlpha,
                        k.alpha,
                        k.spread,
                        z,
                        newPosition,
                        stepPnl,
                        portfolio.pnl
                ));
            }
        }

        System.out.println("========================================================================================================================================");
        System.out.println();
        System.out.println("Model Diagnostics");
        System.out.println("------------------------------------------------------------");
        System.out.printf("%-42s : %12.6f%n", "Average absolute beta estimation error", betaAbsError / 180.0);
        System.out.printf("%-42s : %12.6f%n", "Average absolute alpha estimation error", alphaAbsError / 180.0);
        System.out.printf("%-42s : %12.6f%n", "Average absolute spread", spreadAbs / 180.0);
        System.out.printf("%-42s : %12d%n", "Position changes", trades);
        System.out.printf("%-42s : %12.6f%n", "Final strategy PnL", portfolio.pnl);
        System.out.printf("%-42s : %12.6f%n", "Annualized Sharpe-like ratio", portfolio.sharpe());
        System.out.printf("%-42s : %12.6f%n", "Maximum drawdown", portfolio.maxDrawdown);

        System.out.println();
        System.out.println("Concepts Demonstrated");
        System.out.println("------------------------------------------------------------");
        System.out.println("1. State-space modelling for alpha and beta");
        System.out.println("2. Recursive Kalman update instead of static regression");
        System.out.println("3. Dynamic hedge-ratio estimation");
        System.out.println("4. Spread and rolling z-score construction");
        System.out.println("5. Mean-reversion trading signal generation");
        System.out.println("6. Transaction-cost-adjusted PnL tracking");
        System.out.println("7. Sharpe-like and drawdown diagnostics");
    }
}
