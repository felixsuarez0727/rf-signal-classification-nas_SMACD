"""
Neural Architecture Search (NAS) for wireless signal classification
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
    BatchNormalization, Input, GlobalAveragePooling1D,
    Bidirectional, SeparableConv1D, AveragePooling1D
)
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import random
from typing import Dict, List, Tuple, Any
# Import config from parent directory
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from wireless_classifier_optimized.config import OPTIMIZATION_CONFIG
except ImportError:
    # Fallback configuration if import fails
    OPTIMIZATION_CONFIG = {
        'chunk_size': 512,
        'target_params': 5000,
        'cnn_filters': [16, 32],
        'lstm_units': 32,
        'dense_units': 16,
        'dropout_rate': 0.2,
        'epochs': 15,
        'batch_size': 16,
        'learning_rate': 0.001
    }


class WirelessSignalNAS:
    """
    Neural Architecture Search for wireless signal classification models
    """
    
    def __init__(self, input_shape: Tuple, num_classes: int = 3, 
                 population_size: int = 20, generations: int = 10):
        """
        Initialize NAS for wireless signal classification
        
        Args:
            input_shape: Input tensor shape (time_steps, features)
            num_classes: Number of output classes
            population_size: Number of architectures in population
            generations: Number of evolution generations
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.population_size = population_size
        self.generations = generations
        
        # Architecture search space
        self.search_space = self._define_search_space()
        
        # Population and results tracking
        self.population = []
        self.fitness_history = []
        self.best_architecture = None
        self.best_fitness = float('inf')
        
        print(f"🧬 NAS Initialized:")
        print(f"   Input shape: {input_shape}")
        print(f"   Population size: {population_size}")
        print(f"   Generations: {generations}")
        print(f"   Search space size: {self._calculate_search_space_size()}")
    
    def _define_search_space(self) -> Dict[str, List]:
        """
        Define the search space for wireless signal architectures
        """
        return {
            # CNN Layer configurations - MORE EFFECTIVE
            'conv_layers': [2, 3, 4],  # Remove single layer
            'conv_filters': [[16, 32], [32, 64], [16, 32, 64], [32, 64, 128], [16, 32, 64, 128]],
            'conv_kernels': [3, 5],  # Remove large kernels
            'conv_activation': ['relu', 'elu'],  # Remove swish for stability
            'conv_type': ['standard'],  # Focus on standard conv
            
            # Pooling configurations - BETTER POOLING
            'pooling_type': ['max', 'average'],  # Remove global_avg initially
            'pooling_size': [2, 3],  # Smaller pooling
            
            # LSTM configurations - MORE FOCUSED
            'lstm_layers': [1, 2],  # Remove 0 layers
            'lstm_units': [32, 64, 128],  # Remove very small units
            'lstm_bidirectional': [True, False],
            'lstm_dropout': [0.2, 0.3],  # Higher dropout
            
            # Dense layer configurations - BETTER SIZES
            'dense_layers': [1, 2],  # Remove 3 layers
            'dense_units': [16, 32, 64],  # Remove very small units
            'dense_activation': ['relu', 'elu'],
            'dense_dropout': [0.3, 0.4],  # Higher dropout
            
            # Regularization - STRONGER
            'batch_norm': [True],  # Always use batch norm
            'dropout_rate': [0.2, 0.3, 0.4],  # Higher dropout
            
            # Optimizer configurations - BETTER LEARNING RATES
            'optimizer': ['adam'],  # Focus on adam
            'learning_rate': [0.001, 0.0005],  # Remove very small LR
            'batch_size': [32, 64]  # Remove small batch
        }
    
    def _calculate_search_space_size(self) -> int:
        """
        Calculate total search space size
        """
        total_combinations = 1
        for key, values in self.search_space.items():
            total_combinations *= len(values)
        return total_combinations
    
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """
        Generate a SMART random architecture from search space
        """
        architecture = {}
        
        # Generate architecture with constraints
        for key, values in self.search_space.items():
            if key == 'conv_filters':
                # Match conv_filters to conv_layers
                conv_layers = architecture.get('conv_layers', random.choice(self.search_space['conv_layers']))
                # Choose filter config that matches layer count
                valid_filters = [f for f in values if len(f) >= conv_layers]
                if not valid_filters:
                    valid_filters = values
                architecture[key] = random.choice(valid_filters)
            elif key == 'conv_layers':
                architecture[key] = random.choice(values)
            elif key == 'lstm_units':
                # Always include lstm_units
                architecture[key] = random.choice(values)
            elif key == 'dense_units':
                # Match dense_units to dense_layers
                dense_layers = architecture.get('dense_layers', random.choice(self.search_space['dense_layers']))
                # Choose units that work well together
                if dense_layers == 1:
                    architecture[key] = random.choice([16, 32, 64])
                else:  # 2 layers
                    architecture[key] = random.choice([32, 64])
            elif key == 'dense_layers':
                architecture[key] = random.choice(values)
            else:
                architecture[key] = random.choice(values)
        
        return architecture
    
    def _build_model_from_architecture(self, architecture: Dict[str, Any]) -> tf.keras.Model:
        """
        Build Keras model from architecture specification
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        
        # CNN layers
        conv_layers = architecture.get('conv_layers', 2)
        conv_filters = architecture.get('conv_filters', [16, 32])
        conv_kernel = architecture.get('conv_kernels', 3)
        conv_activation = architecture.get('conv_activation', 'relu')
        conv_type = architecture.get('conv_type', 'standard')
        
        for i in range(conv_layers):
            filters = conv_filters[i] if i < len(conv_filters) else conv_filters[-1]
            
            if conv_type == 'separable':
                model.add(SeparableConv1D(filters, conv_kernel, activation=conv_activation, padding='same'))
            else:
                model.add(Conv1D(filters, conv_kernel, activation=conv_activation, padding='same'))
            
            if architecture.get('batch_norm', True):
                model.add(BatchNormalization())
            
            # Pooling
            pooling_type = architecture.get('pooling_type', 'max')
            pooling_size = architecture.get('pooling_size', 2)
            
            if pooling_type == 'max':
                model.add(MaxPooling1D(pooling_size))
            elif pooling_type == 'average':
                model.add(AveragePooling1D(pooling_size))
            
            model.add(Dropout(architecture.get('dropout_rate', 0.2)))
        
        # Global pooling if specified
        if architecture.get('pooling_type') == 'global_avg':
            model.add(GlobalAveragePooling1D())
        
        # LSTM layers
        lstm_layers = architecture.get('lstm_layers', 1)
        lstm_units = architecture.get('lstm_units', 32)
        lstm_bidirectional = architecture.get('lstm_bidirectional', False)
        lstm_dropout = architecture.get('lstm_dropout', 0.2)
        
        # Only add LSTM if lstm_layers > 0 and we have temporal data
        if lstm_layers > 0 and architecture.get('pooling_type') != 'global_avg':
            for i in range(lstm_layers):
                units = lstm_units[i] if isinstance(lstm_units, list) and i < len(lstm_units) else lstm_units
                
                if lstm_bidirectional:
                    model.add(Bidirectional(LSTM(units, dropout=lstm_dropout, return_sequences=(i < lstm_layers-1))))
                else:
                    model.add(LSTM(units, dropout=lstm_dropout, return_sequences=(i < lstm_layers-1)))
        
        # Dense layers
        dense_layers = architecture.get('dense_layers', 2)
        dense_units = architecture.get('dense_units', 16)
        dense_activation = architecture.get('dense_activation', 'relu')
        dense_dropout = architecture.get('dense_dropout', 0.2)
        
        for i in range(dense_layers):
            units = dense_units[i] if isinstance(dense_units, list) and i < len(dense_units) else dense_units
            model.add(Dense(units, activation=dense_activation))
            model.add(Dropout(dense_dropout))
        
        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile model
        optimizer_name = architecture.get('optimizer', 'adam')
        learning_rate = architecture.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = SGD(learning_rate=learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _evaluate_architecture(self, architecture: Dict[str, Any], 
                             X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             max_epochs: int = 5) -> Dict[str, float]:
        """
        Evaluate architecture fitness
        """
        try:
            # Build model
            model = self._build_model_from_architecture(architecture)
            
            # BETTER training with regularization
            from tensorflow.keras.callbacks import ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(patience=3, restore_best_weights=True, monitor='val_loss'),
                ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-6, monitor='val_loss')
            ]
            
            # Train model with better settings
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=max_epochs,
                batch_size=architecture.get('batch_size', 32),
                callbacks=callbacks,
                verbose=0
            )
            
            # Calculate fitness (IMPROVED multi-objective)
            val_accuracy = max(history.history['val_accuracy'])
            param_count = model.count_params()
            
            # IMPROVED fitness function
            # Penalize low accuracy heavily
            accuracy_penalty = (1 - val_accuracy) ** 2  # Quadratic penalty
            
            # Parameter penalty (target: <10k params)
            if param_count <= 10000:
                param_penalty = 0  # No penalty for small models
            elif param_count <= 20000:
                param_penalty = 0.1 * (param_count - 10000) / 10000
            else:
                param_penalty = 0.1 + 0.2 * (param_count - 20000) / 30000
            
            # Combined fitness (lower is better)
            fitness = accuracy_penalty + param_penalty
            
            return {
                'fitness': fitness,
                'accuracy': val_accuracy,
                'parameters': param_count,
                'model_size_mb': param_count * 4 / (1024 * 1024)
            }
            
        except Exception as e:
            print(f"⚠️ Architecture evaluation failed: {e}")
            return {
                'fitness': float('inf'),
                'accuracy': 0.0,
                'parameters': float('inf'),
                'model_size_mb': float('inf')
            }
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Crossover operation between two parent architectures
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Randomly select genes to crossover
        crossover_keys = random.sample(list(self.search_space.keys()), 
                                     random.randint(1, len(self.search_space.keys()) // 2))
        
        for key in crossover_keys:
            child1[key] = parent2[key]
            child2[key] = parent1[key]
        
        return child1, child2
    
    def _mutate(self, architecture: Dict[str, Any], mutation_rate: float = 0.1) -> Dict[str, Any]:
        """
        Mutate architecture with given mutation rate
        """
        mutated = architecture.copy()
        
        for key, values in self.search_space.items():
            if random.random() < mutation_rate:
                mutated[key] = random.choice(values)
        
        return mutated
    
    def search(self, X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Perform neural architecture search
        """
        print(f"🔍 Starting NAS search...")
        print(f"   Population: {self.population_size}, Generations: {self.generations}")
        
        # Initialize population
        print("1️⃣ Initializing population...")
        self.population = []
        
        for i in range(self.population_size):
            architecture = self._generate_random_architecture()
            fitness = self._evaluate_architecture(architecture, X_train, y_train, X_val, y_val)
            architecture['fitness'] = fitness['fitness']
            architecture['metrics'] = fitness
            self.population.append(architecture)
            
            if i % 5 == 0:
                print(f"   Generated {i+1}/{self.population_size} architectures")
        
        # Evolution loop
        for generation in range(self.generations):
            print(f"\n2️⃣ Generation {generation + 1}/{self.generations}")
            
            # Sort by fitness
            self.population.sort(key=lambda x: x['fitness'])
            
            # Track best
            if self.population[0]['fitness'] < self.best_fitness:
                self.best_fitness = self.population[0]['fitness']
                self.best_architecture = self.population[0].copy()
            
            # Print generation statistics
            best_acc = self.population[0]['metrics']['accuracy']
            best_params = self.population[0]['metrics']['parameters']
            avg_fitness = np.mean([arch['fitness'] for arch in self.population[:10]])
            
            print(f"   Best fitness: {self.best_fitness:.4f}")
            print(f"   Best accuracy: {best_acc:.4f}")
            print(f"   Best parameters: {best_params:,}")
            print(f"   Average fitness (top 10): {avg_fitness:.4f}")
            
            # Create next generation
            new_population = []
            
            # Elite selection (keep best 20%)
            elite_size = self.population_size // 5
            new_population.extend(self.population[:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1, mutation_rate=0.15)
                child2 = self._mutate(child2, mutation_rate=0.15)
                
                # Evaluate children
                fitness1 = self._evaluate_architecture(child1, X_train, y_train, X_val, y_val)
                fitness2 = self._evaluate_architecture(child2, X_train, y_train, X_val, y_val)
                
                child1['fitness'] = fitness1['fitness']
                child1['metrics'] = fitness1
                child2['fitness'] = fitness2['fitness']
                child2['metrics'] = fitness2
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.population_size]
            
            # Store fitness history
            self.fitness_history.append({
                'generation': generation + 1,
                'best_fitness': self.best_fitness,
                'best_accuracy': best_acc,
                'best_parameters': best_params
            })
        
        print(f"\n🎉 NAS Search Completed!")
        print(f"   Best architecture found:")
        print(f"   - Fitness: {self.best_fitness:.4f}")
        print(f"   - Accuracy: {self.best_architecture['metrics']['accuracy']:.4f}")
        print(f"   - Parameters: {self.best_architecture['metrics']['parameters']:,}")
        
        return self.best_architecture
    
    def _tournament_selection(self, tournament_size: int = 3) -> Dict[str, Any]:
        """
        Tournament selection for parent selection
        """
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return min(tournament, key=lambda x: x['fitness'])
    
    def get_search_results(self) -> Dict[str, Any]:
        """
        Get comprehensive search results
        """
        return {
            'best_architecture': self.best_architecture,
            'fitness_history': self.fitness_history,
            'population_size': self.population_size,
            'generations': self.generations,
            'search_space_size': self._calculate_search_space_size(),
            'final_population': sorted(self.population, key=lambda x: x['fitness'])[:10]
        }
    
    def visualize_search_progress(self, save_path: str = None):
        """
        Visualize NAS search progress
        """
        import matplotlib.pyplot as plt
        
        if not self.fitness_history:
            print("⚠️ No search history available")
            return
        
        generations = [h['generation'] for h in self.fitness_history]
        best_fitness = [h['best_fitness'] for h in self.fitness_history]
        best_accuracy = [h['best_accuracy'] for h in self.fitness_history]
        best_parameters = [h['best_parameters'] for h in self.fitness_history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Fitness evolution
        ax1.plot(generations, best_fitness, 'b-', marker='o')
        ax1.set_title('Best Fitness Evolution')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.grid(True)
        
        # Accuracy evolution
        ax2.plot(generations, best_accuracy, 'g-', marker='s')
        ax2.set_title('Best Accuracy Evolution')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)
        
        # Parameter count evolution
        ax3.plot(generations, best_parameters, 'r-', marker='^')
        ax3.set_title('Best Parameter Count Evolution')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Parameters')
        ax3.grid(True)
        
        # Pareto front (Accuracy vs Parameters)
        final_pop = self.get_search_results()['final_population']
        accuracies = [arch['metrics']['accuracy'] for arch in final_pop]
        parameters = [arch['metrics']['parameters'] for arch in final_pop]
        
        ax4.scatter(parameters, accuracies, alpha=0.6)
        ax4.set_title('Accuracy vs Parameters (Final Population)')
        ax4.set_xlabel('Parameters')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Search progress visualization saved: {save_path}")
        
        plt.show()


def run_nas_demo(X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                input_shape: Tuple, num_classes: int = 3):
    """
    Run NAS demonstration for wireless signal classification
    """
    print("🧬 NEURAL ARCHITECTURE SEARCH DEMO")
    print("=" * 50)
    
    # Initialize NAS
    nas = WirelessSignalNAS(
        input_shape=input_shape,
        num_classes=num_classes,
        population_size=15,  # Smaller for demo
        generations=5        # Fewer for demo
    )
    
    # Run search
    best_architecture = nas.search(X_train, y_train, X_val, y_val)
    
    # Get results
    results = nas.get_search_results()
    
    # Visualize progress
    nas.visualize_search_progress('nas_search_progress.png')
    
    # Print detailed results
    print(f"\n📊 NAS SEARCH RESULTS:")
    print(f"   Search space size: {results['search_space_size']:,}")
    print(f"   Architectures evaluated: {results['population_size'] * results['generations']:,}")
    print(f"   Best architecture found:")
    
    best = results['best_architecture']
    print(f"   - Accuracy: {best['metrics']['accuracy']:.4f}")
    print(f"   - Parameters: {best['metrics']['parameters']:,}")
    print(f"   - Model size: {best['metrics']['model_size_mb']:.2f} MB")
    
    # Show architecture details
    print(f"\n🏗️ BEST ARCHITECTURE DETAILS:")
    for key, value in best.items():
        if key not in ['fitness', 'metrics']:
            print(f"   {key}: {value}")
    
    return nas, results


if __name__ == "__main__":
    print("🧬 NAS module for wireless signal classification")
    print("Use run_nas_demo() function to start the search")
