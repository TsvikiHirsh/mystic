"""
mystic_lmfit: An lmfit-compatible API using mystic optimization engine

This module provides an lmfit-style interface for curve fitting and parameter
estimation, powered by mystic's robust optimization solvers and constraint handling.

Users familiar with lmfit can use this module with minimal changes to their code,
while benefiting from mystic's advanced constraint capabilities and global optimization.
"""

import numpy as np
from collections import OrderedDict
from mystic.solvers import DifferentialEvolutionSolver, NelderMeadSimplexSolver, PowellDirectionalSolver
from mystic.solvers import BuckshotSolver, LatticeSolver
from mystic.termination import VTR, ChangeOverGeneration, CandidateRelativeTolerance, Or
from mystic.monitors import Monitor


class Parameter:
    """
    An lmfit-compatible Parameter class.

    Represents a single parameter in a fit with value, bounds, and vary flag.
    """
    def __init__(self, name, value=None, vary=True, min=-np.inf, max=np.inf,
                 expr=None, brute_step=None, user_data=None):
        self.name = name
        self.value = value if value is not None else 0.0
        self.vary = vary
        self.min = min
        self.max = max
        self.expr = expr  # Not implemented in this basic version
        self.brute_step = brute_step
        self.user_data = user_data
        self.stderr = None
        self.correl = None

    def __repr__(self):
        s = f"<Parameter '{self.name}'"
        s += f", value={self.value}"
        if not self.vary:
            s += " (fixed)"
        if np.isfinite(self.min):
            s += f", min={self.min}"
        if np.isfinite(self.max):
            s += f", max={self.max}"
        s += ">"
        return s


class Parameters(OrderedDict):
    """
    An lmfit-compatible Parameters class.

    An ordered dictionary of Parameter objects with convenience methods.
    """
    def add(self, name, value=None, vary=True, min=-np.inf, max=np.inf,
            expr=None, brute_step=None, user_data=None):
        """Add a Parameter."""
        self[name] = Parameter(name, value=value, vary=vary, min=min, max=max,
                              expr=expr, brute_step=brute_step, user_data=user_data)

    def valuesdict(self):
        """Return parameter values as a dictionary."""
        return {name: par.value for name, par in self.items()}

    def pretty_print(self):
        """Print parameters in a nice format."""
        print("Parameters:")
        for name, par in self.items():
            print(f"  {par}")


class ModelResult:
    """
    Result of a model fit, compatible with lmfit's ModelResult.

    Attributes
    ----------
    params : Parameters
        The Parameters object with fitted values
    success : bool
        Whether the fit succeeded
    message : str
        Message about the fit
    nfev : int
        Number of function evaluations
    nvarys : int
        Number of variables parameters
    ndata : int
        Number of data points
    chisqr : float
        Chi-square statistic
    redchi : float
        Reduced chi-square
    residual : ndarray
        Residuals from the fit
    best_values : dict
        Dictionary of parameter names to best-fit values
    """
    def __init__(self, params, success, message, nfev, nvarys, ndata, chisqr,
                 residual, solver=None):
        self.params = params
        self.success = success
        self.message = message
        self.nfev = nfev
        self.nvarys = nvarys
        self.ndata = ndata
        self.chisqr = chisqr
        self.redchi = chisqr / max(1, ndata - nvarys) if ndata > nvarys else np.inf
        self.residual = residual
        self.best_values = params.valuesdict()
        self._solver = solver

    def __repr__(self):
        s = "[[Model Result]]\n"
        s += f"  Success: {self.success}\n"
        s += f"  Message: {self.message}\n"
        s += f"  Chi-square: {self.chisqr:.6e}\n"
        s += f"  Reduced chi-square: {self.redchi:.6e}\n"
        s += f"  Function evaluations: {self.nfev}\n"
        s += f"  Variables: {self.nvarys}\n"
        s += f"  Data points: {self.ndata}\n"
        s += "\n[[Parameters]]\n"
        for name, value in self.best_values.items():
            par = self.params[name]
            vary_str = "" if par.vary else " (fixed)"
            s += f"  {name}: {value:.6e}{vary_str}\n"
        return s


class Model:
    """
    An lmfit-compatible Model class using mystic optimization.

    The Model represents a mathematical function to be fit to data. It uses
    mystic's optimization solvers under the hood.

    Parameters
    ----------
    func : callable
        The model function. Should have signature func(x, param1, param2, ...)
        where x is the independent variable and param1, param2, ... are parameters.
    independent_vars : list of str, optional
        List of independent variable names. Default is ['x'].
    method : str, optional
        Default optimization method to use. Can be overridden in fit().
        Options: 'differential_evolution', 'nelder', 'powell', 'buckshot', 'lattice'
        Default is 'differential_evolution'.
    **kwargs : optional
        Additional keyword arguments for the model.

    Examples
    --------
    >>> def gaussian(x, amp, cen, wid):
    ...     return amp * np.exp(-(x-cen)**2 / (2*wid**2))
    >>>
    >>> model = Model(gaussian)
    >>> params = model.make_params(amp=5, cen=5, wid=1)
    >>> params['amp'].min = 0
    >>> params['cen'].min = 0
    >>> params['cen'].max = 10
    >>> params['wid'].min = 0
    >>>
    >>> result = model.fit(data, params, x=x, method='differential_evolution')
    """
    def __init__(self, func, independent_vars=None, method='differential_evolution', **kwargs):
        self.func = func
        self.independent_vars = independent_vars or ['x']
        self.default_method = method
        self.kwargs = kwargs

        # Extract parameter names from function signature
        import inspect
        sig = inspect.signature(func)
        self.param_names = [name for name in sig.parameters.keys()
                           if name not in self.independent_vars]

    def make_params(self, **kwargs):
        """
        Create a Parameters object for this model.

        Parameters can be specified as:
        - Simple values: param_name=value
        - Dictionaries: param_name={'value': 5, 'min': 0, 'max': 10, 'vary': True}
        - Parameter objects: param_name=Parameter('name', value=5)

        Returns
        -------
        Parameters
            A Parameters object with all model parameters.
        """
        params = Parameters()

        for name in self.param_names:
            if name in kwargs:
                val = kwargs[name]
                if isinstance(val, Parameter):
                    params[name] = val
                elif isinstance(val, dict):
                    params.add(name, **val)
                else:
                    params.add(name, value=val)
            else:
                params.add(name, value=1.0)

        return params

    def fit(self, data, params=None, method=None, weights=None,
            ftol=1e-6, xtol=1e-6, maxfev=None, maxiter=1000,
            nan_policy='raise', **kwargs):
        """
        Fit the model to data using mystic optimization.

        Parameters
        ----------
        data : array-like
            The data to fit.
        params : Parameters, optional
            Parameters object. If None, created from kwargs.
        method : str, optional
            Optimization method. Options:
            - 'differential_evolution': Global optimization (default)
            - 'nelder': Nelder-Mead simplex (local)
            - 'powell': Powell's method (local)
            - 'buckshot': Ensemble of random starts
            - 'lattice': Ensemble on a grid
        weights : array-like, optional
            Weights for each data point.
        ftol : float, optional
            Tolerance for termination by the change of the cost function.
        xtol : float, optional
            Tolerance for termination by the change of the parameters.
        maxfev : int, optional
            Maximum number of function evaluations.
        maxiter : int, optional
            Maximum number of iterations.
        nan_policy : str, optional
            How to handle NaN values ('raise', 'propagate', 'omit').
        **kwargs : optional
            Independent variables (e.g., x=x_data).

        Returns
        -------
        ModelResult
            Object containing the fit results.
        """
        # Create params from kwargs if not provided
        if params is None:
            params = self.make_params()
            for name in self.param_names:
                if name in kwargs:
                    val = kwargs.pop(name)
                    if isinstance(val, (int, float)):
                        params[name].value = val

        # Extract independent variables
        indep_vars = {}
        for var in self.independent_vars:
            if var in kwargs:
                indep_vars[var] = np.asarray(kwargs[var])

        # Convert data to array
        data = np.asarray(data)

        # Handle weights
        if weights is not None:
            weights = np.asarray(weights)

        # Extract parameter info
        param_names = [name for name, par in params.items() if par.vary]
        x0 = np.array([params[name].value for name in param_names])
        lower = np.array([params[name].min for name in param_names])
        upper = np.array([params[name].max for name in param_names])

        # Check for infinite bounds and adjust
        lower = np.where(np.isfinite(lower), lower, -1e10)
        upper = np.where(np.isfinite(upper), upper, 1e10)

        # Create cost function (sum of squared residuals)
        def cost_function(p):
            # Update varying parameters
            for i, name in enumerate(param_names):
                params[name].value = p[i]

            # Evaluate model
            model_values = self.func(**indep_vars, **params.valuesdict())

            # Calculate residuals
            residuals = data - model_values

            # Handle NaN
            if nan_policy == 'raise' and np.any(np.isnan(residuals)):
                raise ValueError("NaN values detected in residuals")
            elif nan_policy == 'omit':
                residuals = residuals[~np.isnan(residuals)]

            # Apply weights if provided
            if weights is not None:
                residuals = residuals * weights

            # Return sum of squared residuals
            return np.sum(residuals**2)

        # Select method
        if method is None:
            method = self.default_method

        # Run optimization
        solver = self._create_solver(method, len(x0))

        # Set initial points (not supported by ensemble solvers)
        if not getattr(solver, '_is_ensemble', False):
            solver.SetInitialPoints(x0)

        # Set bounds
        solver.SetStrictRanges(lower, upper)

        # Set termination conditions
        term = Or([
            ChangeOverGeneration(ftol, generations=50),
            CandidateRelativeTolerance(xtol, ftol)
        ])
        solver.SetTermination(term)

        # Set evaluation limits
        if maxfev is None:
            maxfev = maxiter * 1000
        solver.SetEvaluationLimits(iterations=maxiter, evaluations=maxfev)

        # Set up monitoring
        monitor = Monitor()
        solver.SetGenerationMonitor(monitor)

        # Solve
        solver.Solve(cost_function, disp=False)

        # Update parameters with results
        best_params = solver.bestSolution
        for i, name in enumerate(param_names):
            params[name].value = best_params[i]

        # Calculate final residuals
        model_values = self.func(**indep_vars, **params.valuesdict())
        residuals = data - model_values
        if weights is not None:
            weighted_residuals = residuals * weights
            chisqr = np.sum(weighted_residuals**2)
        else:
            chisqr = np.sum(residuals**2)

        # Create result object
        result = ModelResult(
            params=params,
            success=solver.Terminated(),
            message=f"Optimization terminated successfully using {method}",
            nfev=solver.evaluations,
            nvarys=len(param_names),
            ndata=len(data),
            chisqr=chisqr,
            residual=residuals,
            solver=solver
        )

        return result

    def _create_solver(self, method, dim):
        """Create a mystic solver based on the method name."""
        method = method.lower()

        if method in ['differential_evolution', 'de']:
            npop = min(dim * 10, 40)  # Population size
            solver = DifferentialEvolutionSolver(dim, npop)
            solver._is_ensemble = False
            return solver

        elif method in ['nelder', 'neldermead', 'nelder-mead']:
            solver = NelderMeadSimplexSolver(dim)
            solver._is_ensemble = False
            return solver

        elif method == 'powell':
            solver = PowellDirectionalSolver(dim)
            solver._is_ensemble = False
            return solver

        elif method == 'buckshot':
            # Ensemble solver with random starts
            npts = 8  # Number of solver instances
            solver = BuckshotSolver(dim, npts)
            solver.SetNestedSolver(NelderMeadSimplexSolver)
            solver._is_ensemble = True
            return solver

        elif method == 'lattice':
            # Ensemble solver on a grid
            npts = 8  # Number of solver instances
            solver = LatticeSolver(dim, npts)
            solver.SetNestedSolver(NelderMeadSimplexSolver)
            solver._is_ensemble = True
            return solver

        else:
            raise ValueError(f"Unknown method: {method}. "
                           f"Choose from: differential_evolution, nelder, powell, buckshot, lattice")


def create_params(**kwargs):
    """
    Create a Parameters object from keyword arguments.

    This is a convenience function for creating Parameters objects.

    Parameters
    ----------
    **kwargs : optional
        Parameter specifications. Each can be:
        - A simple value: param_name=value
        - A dictionary: param_name={'value': 5, 'min': 0, 'max': 10}

    Returns
    -------
    Parameters
        A Parameters object.

    Examples
    --------
    >>> params = create_params(
    ...     amp={'value': 10, 'min': 0},
    ...     center=5,
    ...     width={'value': 1, 'min': 0, 'vary': True}
    ... )
    """
    params = Parameters()

    for name, val in kwargs.items():
        if isinstance(val, Parameter):
            params[name] = val
        elif isinstance(val, dict):
            params.add(name, **val)
        else:
            params.add(name, value=val)

    return params
