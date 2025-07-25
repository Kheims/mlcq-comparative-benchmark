"""
Evolutionary algorithms for code smell detection.
Includes Genetic Programming classifier and evolutionary feature selection.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score
import joblib
import logging
import random
from collections import Counter

from deap import base, creator, tools, algorithms, gp
import operator
import math

logger = logging.getLogger(__name__)


class GeneticProgrammingClassifier(BaseEstimator, ClassifierMixin):
    """
    Genetic Programming classifier for code smell detection.
    Uses DEAP library for evolutionary computation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GP classifier.
        
        Args:
            config: Configuration dictionary with GP parameters
        """
        self.config = config
        self.population_size = config.get('population_size', 100)
        self.generations = config.get('generations', 50)
        self.tournament_size = config.get('tournament_size', 3)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.max_depth = config.get('max_depth', 10)
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_individual = None
        self.is_trained = False
        self.feature_names = None
        self.classes_ = None
        
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
    
    def _setup_gp_primitives(self, n_features: int):
        """Set up GP primitives and terminal set."""
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        self.pset = gp.PrimitiveSet("MAIN", n_features)
        
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(self._protected_div, 2)
        self.pset.addPrimitive(operator.neg, 1)
        self.pset.addPrimitive(abs, 1)
        self.pset.addPrimitive(max, 2)
        self.pset.addPrimitive(min, 2)
        self.pset.addPrimitive(self._protected_log, 1)
        self.pset.addPrimitive(self._protected_sqrt, 1)
        
        self.pset.addPrimitive(self._if_then_else, 3)
        self.pset.addPrimitive(self._greater_than, 2)
        self.pset.addPrimitive(self._less_than, 2)
        
        self.pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
        
        if self.feature_names is not None:
            for i, name in enumerate(self.feature_names[:n_features]):
                self.pset.renameArguments(**{f'ARG{i}': f'F{i}_{name[:10]}'})
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, 
                             min_=1, max_=self.max_depth)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                             self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)
        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr_mut, 
                             pset=self.pset)
        
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), 
                                                    max_value=self.max_depth))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), 
                                                      max_value=self.max_depth))
    
    def _protected_div(self, left, right):
        """Protected division to avoid division by zero."""
        if abs(right) < 1e-6:
            return 1.0
        return left / right
    
    def _protected_log(self, x):
        """Protected logarithm to avoid log of negative numbers."""
        if x <= 0:
            return 0.0
        return math.log(abs(x))
    
    def _protected_sqrt(self, x):
        """Protected square root."""
        return math.sqrt(abs(x))
    
    def _if_then_else(self, condition, if_true, if_false):
        """If-then-else operator."""
        return if_true if condition > 0 else if_false
    
    def _greater_than(self, left, right):
        """Greater than comparison."""
        return 1.0 if left > right else 0.0
    
    def _less_than(self, left, right):
        """Less than comparison."""
        return 1.0 if left < right else 0.0
    
    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual."""
        func = self.toolbox.compile(expr=individual)
        
        predictions = []
        for i in range(len(self.X_train)):
            try:
                features = self.X_train[i]
                output = func(*features)
                prediction = 1 if output > 0 else 0
                predictions.append(prediction)
            except (ValueError, ZeroDivisionError, OverflowError, 
                    TypeError, ArithmeticError):
                predictions.append(0)
        
        try:
            fitness = f1_score(self.y_train, predictions, average='weighted', 
                             zero_division=0.0)
        except:
            fitness = 0.0
        
        return (fitness,)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the GP classifier.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Training Genetic Programming classifier...")
        
        self.X_train = self.scaler.fit_transform(X)
        self.y_train = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        
        self._setup_gp_primitives(X.shape[1])
        
        population = self.toolbox.population(n=self.population_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logger.info(f"Starting evolution with {self.population_size} individuals "
                   f"for {self.generations} generations...")
        
        population, logbook = algorithms.eaSimple(
            population, self.toolbox, 
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=stats,
            verbose=False
        )
        
        best_fitness = 0
        for ind in population:
            if ind.fitness.values[0] > best_fitness:
                best_fitness = ind.fitness.values[0]
                self.best_individual = ind
        
        logger.info(f"Best fitness achieved: {best_fitness:.4f}")
        logger.info(f"Best individual: {str(self.best_individual)}")
        
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Test features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        func = self.toolbox.compile(expr=self.best_individual)
        
        predictions = []
        for i in range(len(X_scaled)):
            try:
                features = X_scaled[i]
                output = func(*features)
                prediction = 1 if output > 0 else 0
                predictions.append(prediction)
            except:
                predictions.append(0)
        
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Test features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        func = self.toolbox.compile(expr=self.best_individual)
        
        raw_outputs = []
        for i in range(len(X_scaled)):
            try:
                features = X_scaled[i]
                output = func(*features)
                raw_outputs.append(output)
            except:
                raw_outputs.append(0.0)
        
        probabilities = np.array(raw_outputs)
        probabilities = 1 / (1 + np.exp(-probabilities))
        
        prob_matrix = np.zeros((len(probabilities), len(self.classes_)))
        for i, prob in enumerate(probabilities):
            prob_matrix[i, 0] = 1 - prob  
            prob_matrix[i, 1] = prob              
        return prob_matrix
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def get_best_individual_str(self) -> str:
        """Get string representation of best individual."""
        if self.best_individual is None:
            return "No individual found"
        return str(self.best_individual)
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'best_individual': self.best_individual,
            'classes_': self.classes_,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"GP model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
        """
        model_data = joblib.load(filepath)
        
        self.config = model_data['config']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.best_individual = model_data['best_individual']
        self.classes_ = model_data['classes_']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        n_features = len(self.scaler.mean_) if hasattr(self.scaler, 'mean_') else 10
        self._setup_gp_primitives(n_features)
        
        logger.info(f"GP model loaded from {filepath}")


class EvolutionaryFeatureSelector:
    """
    Evolutionary algorithm for feature selection.
    Uses genetic algorithm to find optimal feature subsets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evolutionary feature selector.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.population_size = config.get('population_size', 50)
        self.generations = config.get('generations', 30)
        self.tournament_size = config.get('tournament_size', 3)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.8)
        self.random_state = config.get('random_state', 42)
        
        self.selected_features = None
        self.best_fitness = 0
        
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        
    
    def _setup_feature_selection_ga(self, n_features: int):
        """Set up GA for feature selection."""
        if hasattr(creator, 'FitnessMax'):
            del creator.FitnessMax
        if hasattr(creator, 'Individual'):
            del creator.Individual
        
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_bool, n_features)
        self.toolbox.register("population", tools.initRepeat, list, 
                             self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate_feature_subset)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
    
    def _evaluate_feature_subset(self, individual):
        """Evaluate a feature subset."""
        feature_mask = np.array(individual, dtype=bool)
        
        if not np.any(feature_mask):
            return (0.0,)
        
        X_subset = self.X_train[:, feature_mask]
        
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=10, random_state=self.random_state)
        
        try:
            scores = cross_val_score(clf, X_subset, self.y_train, cv=3, 
                                   scoring='f1_weighted')
            fitness = np.mean(scores)
        except:
            fitness = 0.0
        
        return (fitness,)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Find optimal feature subset.
        
        Args:
            X: Training features
            y: Training labels
        """
        logger.info("Starting evolutionary feature selection...")
        
        self.X_train = X
        self.y_train = y
        
        self._setup_feature_selection_ga(X.shape[1])
        
        population = self.toolbox.population(n=self.population_size)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=self.crossover_rate,
            mutpb=self.mutation_rate,
            ngen=self.generations,
            stats=stats,
            verbose=False
        )
        
        best_individual = tools.selBest(population, 1)[0]
        self.selected_features = np.array(best_individual, dtype=bool)
        self.best_fitness = best_individual.fitness.values[0]
        
        logger.info(f"Selected {np.sum(self.selected_features)} features "
                   f"out of {len(self.selected_features)}")
        logger.info(f"Best fitness: {self.best_fitness:.4f}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using selected subset.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if self.selected_features is None:
            raise ValueError("Feature selector must be fitted before transforming")
        
        return X[:, self.selected_features]
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit feature selector and transform features.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Transformed features
        """
        self.fit(X, y)
        return self.transform(X)
    
    def get_selected_features(self) -> np.ndarray:
        """Get indices of selected features."""
        if self.selected_features is None:
            return np.array([])
        return np.where(self.selected_features)[0]

