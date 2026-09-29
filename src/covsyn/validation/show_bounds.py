import numpy as np

p = np.load('./variable/course_parameters.npy')
lb = np.load('./variable/course_parameters_lb.npy')
ub = np.load('./variable/course_parameters_ub.npy')
DAYS = {'household': (70, 2.00), 'school': (95, 2.95), 'workplace': (120, 4.02),
        'health_care': (145, 1.99), 'municipality': (170, 3.91)}
print('%-13s %10s %-22s %10s %s' % ('layer', 'daily', '[lb, ub]', 'cumul %', '[lb, ub]'))
for nm, (i, n) in DAYS.items():
    j = i - 37
    d, dl, du = (float(np.mean(x[j:j + 25])) for x in (p, lb, ub))
    f = lambda x: 100 * (1 - (1 - x) ** n)
    print('%-13s %10.5f [%.5f, %.5f] %9.2f  [%.2f, %.2f]'
          % (nm, d, dl, du, f(d), f(dl), f(du)))
print('\nage risk ratios %s  locked=%s' % (np.round(p[26:30], 3).tolist(), bool(np.all(lb[26:30] == ub[26:30]))))
print('latent mean %.2f [%.2f, %.2f]' % (p[0] * p[1], lb[0] * lb[1], ub[0] * ub[1]))
