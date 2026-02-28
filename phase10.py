import numpy as np

# =====================================================
# CONTROL POLICY (Baseline)
# =====================================================
def control_policy(num_stores, num_skus):
    return np.random.uniform(-0.05, 0.05, (num_stores, num_skus))


# =====================================================
# TREATMENT POLICY (RL Simulated)
# =====================================================
def treatment_policy(num_stores, num_skus):
    return np.random.uniform(-0.02, 0.02, (num_stores, num_skus))


# =====================================================
# A/B TEST
# =====================================================
def ab_test():

    num_stores = 10
    num_skus = 425

    control = control_policy(num_stores, num_skus)
    treatment = treatment_policy(num_stores, num_skus)

    control_perf = np.mean(control)
    treatment_perf = np.mean(treatment)

    uplift = treatment_perf - control_perf

    print("Control Performance:", control_perf)
    print("Treatment Performance:", treatment_perf)
    print("Uplift:", uplift)


if __name__ == "__main__":
    ab_test()