def get_gamma_results(cv_results, gamma):
    gamma_results = []

    for g in gamma:
        gamma_results.append(cv_results[cv_results['param_gamma']== g])
    
    return gamma_results