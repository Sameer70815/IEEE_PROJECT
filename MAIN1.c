#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>

#define timestep 0.001
#define T 10
#define printstep 10
#define Njoint 2
#define Ninput 13
#define Nhidden 16
#define Noutput 2

void Create_desired();
void Compute_traj_filter_error();
void Compute_robust_term();
void rk4(void (*kkim)(double x[],double tau[],double xdot[]),
int order, double x_in[],double u[],double x_out[],double t0,double tf);
void robot_dyn(double x[],double tau[],double xdot[]);
void weight_v_dyn(double x[],double tau[],double xdot[]);
void weight_w_dyn(double x[],double tau[],double xdot[]);
void Initialize_sate_weight();
void Input_neural();
void Weight_update_v();
void Weight_update_w();
void HiddeNoutput();
void Neural_output();
void Uniform(double a, double b, int Ninterval, double average[]);
int rad_to_servo_deg(double rad);


double lam=5., Kv=20.;
double Kz=-2.0, Kzl=-15.0, W_max=5;
double t=0.;
double kk=.01;
double F=5.0, G=5.0;

double qd[Njoint+1], qdp[Njoint+1], qdpp[Njoint+1], robot_x[2*Njoint+1];
double e[Njoint+1], ep[Njoint+1], r[Njoint+1], norm_r;
double v_t[Njoint+1], tau[Njoint+1];
double norm_weight;

double input_x[Ninput+1], weight_v[Ninput*Nhidden+1], weight_w[Nhidden*Noutput+1];
double hiddeNnet[Nhidden+1], hiddeNoutput[Nhidden+1], alpha[Nhidden+1], neural_output[Noutput+1];
double tau_weight_v[Ninput*Nhidden+1], tau_weight_w[Nhidden*Noutput+1];

int neural_init=0;

int main()
{
    int i, k=10;
    FILE *frobot_x, *fwei_v, *fwei_w;

    frobot_x = fopen("c:\\work\\x_robot", "w");
    fwei_v = fopen("c:\\work\\x_wei_v", "w");
    fwei_w = fopen("c:\\work\\x_wei_w", "w");
    Initialize_sate_weight();
    t = 0.;
    do {
        Create_desired();
        Compute_traj_filter_error();

        if (neural_init == 1) {
            Input_neural();
            HiddeNoutput();
            Weight_update_v();
            Weight_update_w();
            HiddeNoutput();
            Neural_output();
        }

        Compute_robust_term();

        for (i = 1; i <= Njoint; i++)
            tau[i] = r[i]*Kv + neural_output[i] - v_t[i];

        if (k % printstep == 0) {
            k = 0;
            int elbow_deg = rad_to_servo_deg(robot_x[1]);
            int wrist_deg = rad_to_servo_deg(robot_x[2]);

// Convert desired angles (qd) to degrees as well
            int elbow_qd_deg = rad_to_servo_deg(qd[1]);
            int wrist_qd_deg = rad_to_servo_deg(qd[2]);

            fprintf(frobot_x, "%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %d %d %d %d\n",
             t, e[1], e[2], ep[1], ep[2], neural_output[1], neural_output[2],
             r[1], r[2], qd[1], qd[2], robot_x[1], robot_x[2], tau[1], tau[2],
             elbow_qd_deg, wrist_qd_deg, elbow_deg, wrist_deg);

             printf("Time: %.3f Elbow Angle (deg): %d Wrist Angle (deg): %d\n", t,elbow_deg,wrist_deg);

            fprintf(fwei_v, "%f %f %f %f %f %f\n",
                    weight_v[21], weight_v[60], weight_v[22], weight_v[61],
                    hiddeNoutput[8], hiddeNoutput[12]);
            fprintf(fwei_w, "%f %f %f %f %f %f\n",
                    weight_w[1], weight_w[17], weight_w[5],
                    weight_w[21], weight_w[10], weight_w[26]);
        }
        k++;

        rk4(robot_dyn, 2 * Njoint, robot_x, tau, robot_x, t, t + timestep);
        t += timestep;
    } while (t <= T);

    fclose(frobot_x);
    fclose(fwei_v);
    fclose(fwei_w);
    system("notepad C:\\work\\x_robot");
    system("notepad C:\\work\\x_wei_v");
    system("notepad C:\\work\\x_wei_w");
    return 0;
}

// (Previous content preserved here...)

void robot_dyn(double x[], double tau[], double xdot[])
{
    int i;
    double m11, m12, m22, v1, v2, a1, a2, g1, g2, det;
    double m1 = 0.8, m2 = 2.3, l1 = 1.0, l2 = 1.0, g = 9.8;

    m11 = (m1 + m2) * l1 * l1 + m2 * l2 * l2 + 2.0 * m2 * l1 * l2 * cos(x[2]);
    m22 = m2 * l2 * l2;
    m12 = m22 + m2 * l1 * l2 * cos(x[2]);

    v1 = -m2 * l1 * l2 * (2.0 * x[3] * x[4] + x[4] * x[4]) * sin(x[2]);
    v2 = m2 * l1 * l2 * x[3] * x[3] * sin(x[2]);

    g1 = (m1 + m2) * g * l1 * cos(x[1]) + m2 * g * l2 * cos(x[1] + x[2]);
    g2 = m2 * g * l2 * cos(x[1] + x[2]);

    det = m11 * m22 - m12 * m12;

    a1 = tau[1] - v1 - g1;
    a2 = tau[2] - v2 - g2;

    xdot[1] = x[3];
    xdot[2] = x[4];
    xdot[3] = (m22 * a1 - m12 * a2) / det;
    xdot[4] = (m11 * a2 - m12 * a1) / det;

    if (t == 5) {
        xdot[1] = x[3];
        xdot[2] = x[4];
        xdot[3] = (m22 * a1 - m12 * a2) / det - 4;
        xdot[4] = (m11 * a2 - m12 * a1) / det - 1.6;
    }
}
// (Previous content preserved here...)

void rk4(void (*kkim)(double x[], double tau[], double xdot[]), int order, double x_in[], double u[], double x_out[], double t0, double tf)
{
    int i;
    double h = tf - t0;
    double h2 = h / 2.0;
    double h3 = h / 3.0;
    double h6 = h / 6.0;
    double *x_mid1, *x_mid2, *x_end, *xdot_init, *xdot_mid1, *xdot_mid2, *xdot_end;

    x_mid1 = calloc(order + 1, sizeof(double));
    x_mid2 = calloc(order + 1, sizeof(double));
    x_end = calloc(order + 1, sizeof(double));
    xdot_init = calloc(order + 1, sizeof(double));
    xdot_mid1 = calloc(order + 1, sizeof(double));
    xdot_mid2 = calloc(order + 1, sizeof(double));
    xdot_end = calloc(order + 1, sizeof(double));

    (*kkim)(x_in, u, xdot_init);
    for (i = 0; i < order; i++) x_mid1[i] = x_in[i] + h2 * xdot_init[i];
    (*kkim)(x_mid1, u, xdot_mid1);
    for (i = 0; i < order; i++) x_mid2[i] = x_in[i] + h2 * xdot_mid1[i];
    (*kkim)(x_mid2, u, xdot_mid2);
    for (i = 0; i < order; i++) x_end[i] = x_in[i] + h * xdot_mid2[i];
    (*kkim)(x_end, u, xdot_end);
    for (i = 0; i < order; i++) {
        x_out[i] = x_in[i] + h6 * (xdot_init[i] + xdot_end[i]) + h3 * (xdot_mid1[i] + xdot_mid2[i]);
    }

    free(x_mid1);
    free(x_mid2);
    free(x_end);
    free(xdot_init);
    free(xdot_mid1);
    free(xdot_mid2);
    free(xdot_end);
}  

void Input_neural()
{
    input_x[1] = 1;
    input_x[2] = e[2];
    input_x[3] = qd[1];
    input_x[4] = qd[2];
    input_x[5] = ep[1];
    input_x[6] = ep[2];
    input_x[7] = qdpp[1];
    input_x[8] = qdpp[2];
    input_x[9] = qdp[1];
    input_x[10] = qdp[2];
    input_x[11] = e[1];
    input_x[12] = r[1];
    input_x[13] = r[2];
}

void HiddeNoutput()
{
    int i, j, offset;
    double net;
    for (j = 1; j <= Nhidden; j++) {
        net = 0.0;
        for (i = 1; i <= Ninput; i++) {
            offset = (j - 1) * Ninput + i;
            net += input_x[i] * weight_v[offset];
        }
        hiddeNoutput[j] = 1.0 / (1.0 + exp(-alpha[j] * net));
        hiddeNnet[j] = net;
    }
}

void Neural_output()
{
    int i, j, offset;
    for (j = 1; j <= Noutput; j++) {
        neural_output[j] = 0.0;
        for (i = 1; i <= Nhidden; i++) {
            offset = (j - 1) * Nhidden + i;
            neural_output[j] += hiddeNoutput[i] * weight_w[offset];
        }
    }
}

void Weight_update_w()
{
    int i, j, offset;
    double sigp;
    for (j = 1; j <= Noutput; j++) {
        for (i = 1; i <= Nhidden; i++) {
            offset = (j - 1) * Nhidden + i;
            sigp = (1 - hiddeNoutput[i]) * hiddeNoutput[i] * alpha[i];
            tau_weight_w[offset] = F * hiddeNoutput[i] * r[j] - F * sigp * hiddeNnet[i] * r[j];
        }
    }
    rk4(weight_w_dyn, Noutput * Nhidden, weight_w, tau_weight_w, weight_w, t, t + timestep);
}

void Weight_update_v()
{
    int i, j, k, offset;
    double sigp, alphaval;
    for (j = 1; j <= Nhidden; j++) {
        sigp = (1 - hiddeNoutput[j]) * hiddeNoutput[j];
        alphaval = 0.0;
        offset = j;
        for (i = 1; i <= Noutput; i++) {
            alphaval = weight_w[offset] * r[i];
            offset += Nhidden;
        }
        for (k = 1; k <= Ninput; k++) {
            offset = (j - 1) * Ninput + k;
            tau_weight_v[offset] = G * sigp * alphaval * input_x[k];
        }
    }
    rk4(weight_v_dyn, Nhidden * Ninput, weight_v, tau_weight_v, weight_v, t, t + timestep);
}

void weight_v_dyn(double xx[], double tau[], double xdot[])
{
    int i, j, offset;
    for (i = 1; i <= Nhidden; i++) {
        for (j = 1; j <= Ninput; j++) {
            offset = (i - 1) * Ninput + j;
            xdot[offset] = -kk * G * norm_r * xx[offset] + tau[offset];
        }
    }
}

void weight_w_dyn(double xx[], double tau[], double xdot[])
{
    int i, j, offset;
    for (i = 1; i <= Noutput; i++) {
        for (j = 1; j <= Nhidden; j++) {
            offset = (i - 1) * Nhidden + j;
            xdot[offset] = -kk * F * norm_r * xx[offset] + tau[offset];
        }
    }
}

void Create_desired()
{
    double amp = 1.0, tp = 2.0;
    double fact = 2.0 / tp;
    qd[1] = amp * sin(fact * t);
    qd[2] = amp * cos(fact * t);
    qdp[1] = amp * fact * cos(fact * t);
    qdp[2] = -amp * fact * sin(fact * t);
    qdpp[1] = -amp * fact * fact * sin(fact * t);
    qdpp[2] = -amp * fact * fact * cos(fact * t);
}

void Compute_traj_filter_error()
{
    int i;
    norm_r = 0.0;
    for (i = 1; i <= Njoint; i++) {
        e[i] = qd[i] - robot_x[i];
        ep[i] = qdp[i] - robot_x[i + 2];
        r[i] = ep[i] + lam * e[i];
        norm_r += r[i] * r[i];
    }
    norm_r = sqrt(norm_r);
}

void Compute_robust_term()
{
    int i;
    norm_weight = 0.0;
    for (i = 1; i <= Ninput * Nhidden; i++) {
        norm_weight += weight_v[i] * weight_v[i];
    }
    for (i = 1; i <= Nhidden * Noutput; i++) {
        norm_weight += weight_w[i] * weight_w[i];
    }
    norm_weight = sqrt(norm_weight);

    for (i = 1; i <= Njoint; i++) {
        if (i == 0) Kz = Kzl;
        v_t[i] = -Kz * (norm_weight + W_max) * r[i];
    }
}

void Initialize_sate_weight()
{
    int i;
    double a = 0.5, b = 1.5, a_v = -1.0, b_v = 1.0;
    double *average;

    for (i = 1; i <= 2 * Njoint; i++) robot_x[i] = 0.0;
    robot_x[1] = 1.0;

    for (i = 1; i <= Ninput * Nhidden; i++) {
        weight_v[i]=0;
    }

    for (i = 1; i <= Nhidden * Noutput; i++) {
        weight_w[i] = a_v + (b_v - a_v) *((double)rand() / RAND_MAX);
        printf("%f\n", weight_w[i]);
    }

    average = calloc(Nhidden + 1, sizeof(double));
    Uniform(a, b, Nhidden, average);
    for (i = 1; i <= Nhidden; i++) {
        alpha[i] = average[i];
        alpha[i] = (double)rand() / 100.0;
        alpha[i] = 1.0;
    }
    free(average);
}

void Uniform(double a, double b, int Ninterval, double average[])
{
    int i, j, n = 100;
    double random_number, _uniform_data, delta;
    double *sum_interval, *no_interval;

    delta = (b - a) / Ninterval;
    sum_interval = calloc(Ninterval + 1, sizeof(double));
    no_interval = calloc(Ninterval + 1, sizeof(double));

    for (j = 1; j <= Ninterval; j++) sum_interval[j] = no_interval[j] = 0;

    for (i = 1; i <= n; i++) {
        random_number = (double)rand()/ RAND_MAX;
        _uniform_data = a + (b - a) * random_number;
        for (j = 1; j <= Ninterval; j++) {
            if (_uniform_data >= a + (j - 1) * delta && _uniform_data < a + j * delta) {
                sum_interval[j] += _uniform_data;
                no_interval[j]++;
            }
        }
    }
    for (j = 1; j <= Ninterval; j++) {
        if (no_interval[j] == 0) average[j] = 0;
        else average[j] = sum_interval[j] / no_interval[j];
    }
    free(sum_interval);
    free(no_interval);
}
int rad_to_servo_deg(double rad) {
    
    // Assume rad ∈ [-π/2, π/2]
    // Map: -π/2 → 0, 0 → 90, +π/2 → 180
    double min_rad = -M_PI / 2.0;
    double max_rad = M_PI / 2.0;

    // Normalize: 0 to 1
    double normalized = (rad - min_rad) / (max_rad - min_rad);  // ∈ [0, 1]
    
    // Scale to 0–180
    int deg = (int)(normalized * 180.0);
    
    // Clamp just in case
    if (deg < 0) deg = 0;
    if (deg > 180) deg = 180;

    return deg;
}