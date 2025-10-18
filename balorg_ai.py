#!/usr/bin/env python3
"""
Balorg AI - Advanced Technical Discussion System

This module provides an interactive AI assistant focused on technical
discussions about deep learning, model optimization, and AI applications.
"""

import sys
import random


class BalorgAI:
    """
    Advanced AI technical discussion system.
    
    Responds to trigger phrases with detailed technical information on:
    - Deep Learning Architectures
    - Model Optimization
    - AI Applications
    """
    
    def __init__(self):
        """Initialize the Balorg AI system with knowledge modules."""
        self.trigger_phrase = "My Balorg is AI!"
        self.deep_learning_topics = self._load_deep_learning_knowledge()
        self.optimization_topics = self._load_optimization_knowledge()
        self.application_topics = self._load_application_knowledge()
    
    def _load_deep_learning_knowledge(self):
        """Load deep learning architecture knowledge."""
        return {
            "transformers": {
                "title": "Transformer Architectures",
                "content": """
Transformer Models: State-of-the-Art Architecture

Core Components:
1. Self-Attention Mechanism:
   - Query (Q), Key (K), Value (V) matrices
   - Attention(Q,K,V) = softmax(QK^T/√d_k)V
   - Allows parallel processing of sequences
   
2. Multi-Head Attention:
   - Multiple attention heads capture different relationships
   - Concatenated outputs provide rich representations
   - head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
   
3. Position Encoding:
   - PE(pos, 2i) = sin(pos/10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
   - Injects sequence order information

Advanced Variants:
- BERT: Bidirectional encoder for understanding
- GPT: Autoregressive decoder for generation
- T5: Text-to-Text Transfer Transformer
- Vision Transformers (ViT): Applied to image patches
- Sparse Transformers: O(n√n) complexity reduction

Key Advantages:
• Parallel processing vs sequential RNNs
• Long-range dependency capture
• Transfer learning effectiveness
• Scalability to massive datasets
"""
            },
            "neural_networks": {
                "title": "Neural Network Architectures",
                "content": """
Advanced Neural Network Designs

Convolutional Neural Networks (CNNs):
- Spatial hierarchies through convolution
- Weight sharing reduces parameters
- Translation invariance
- Modern architectures: ResNet, EfficientNet, DenseNet

Recurrent Neural Networks (RNNs):
- Sequential data processing
- Hidden state h_t = f(W_hh·h_{t-1} + W_xh·x_t)
- Variants: LSTM, GRU for vanishing gradient mitigation
- Bidirectional RNNs for context-aware processing

Graph Neural Networks (GNNs):
- Message passing on graph structures
- h_v^{(k+1)} = UPDATE(h_v^{(k)}, AGGREGATE({h_u^{(k)}, u ∈ N(v)}))
- Applications: social networks, molecules, knowledge graphs

Attention Mechanisms:
- Dynamic focus on relevant inputs
- α_i = exp(score(h_t, h_i)) / Σ_j exp(score(h_t, h_j))
- Context vector: c_t = Σ_i α_i h_i

Emerging Architectures:
• Neural Architecture Search (NAS)
• Capsule Networks
• Neural ODEs
• Spiking Neural Networks
"""
            },
            "activation_functions": {
                "title": "Activation Functions & Regularization",
                "content": """
Non-linearities and Regularization Techniques

Activation Functions:
1. ReLU: f(x) = max(0, x)
   - Mitigates vanishing gradients
   - Computationally efficient
   - Dead neuron problem

2. Leaky ReLU: f(x) = max(αx, x) where α ≈ 0.01
   - Prevents dead neurons
   
3. GELU: f(x) = x·Φ(x) (Gaussian Error Linear Unit)
   - Used in BERT, GPT
   - Smooth, non-monotonic
   
4. Swish/SiLU: f(x) = x·sigmoid(βx)
   - Self-gated activation
   - Better performance in deep networks

Regularization:
- Dropout: Randomly zero neurons during training
- Batch Normalization: Normalize layer inputs
- Layer Normalization: Per-sample normalization
- Weight Decay: L2 regularization on parameters
- Data Augmentation: Synthetic training diversity
"""
            }
        }
    
    def _load_optimization_knowledge(self):
        """Load model optimization knowledge."""
        return {
            "quantization": {
                "title": "Model Quantization Techniques",
                "content": """
Quantization: Reducing Model Precision

Types of Quantization:
1. Post-Training Quantization (PTQ):
   - INT8 quantization: 4x memory reduction
   - Minimal accuracy loss (< 1% typically)
   - Fast deployment without retraining
   
2. Quantization-Aware Training (QAT):
   - Simulates quantization during training
   - Better accuracy vs PTQ
   - Forward: quantized, Backward: full precision
   
3. Dynamic Quantization:
   - Activations quantized at runtime
   - Weights statically quantized
   - Good for NLP models (LSTM, Transformers)

Quantization Schemes:
- Symmetric: [-127, 127] → [-α, α]
- Asymmetric: [0, 255] → [β, γ]
- Per-channel vs per-tensor quantization

Mathematical Formulation:
q = clamp(round(x/s) + z, q_min, q_max)
x_q = s(q - z)

Where:
- s: scale factor
- z: zero point
- q: quantized value

Advanced Techniques:
• Mixed-precision quantization
• Binary/Ternary networks
• Knowledge distillation + quantization
• Hardware-aware quantization
"""
            },
            "pruning": {
                "title": "Neural Network Pruning",
                "content": """
Pruning: Structured Sparsity for Efficiency

Pruning Strategies:
1. Magnitude-based Pruning:
   - Remove weights with |w| < threshold
   - Simple and effective baseline
   - Can achieve 90%+ sparsity
   
2. Structured Pruning:
   - Remove entire channels/filters
   - Hardware-friendly acceleration
   - Better for actual deployment speedup
   
3. Lottery Ticket Hypothesis:
   - Sparse subnetworks exist at initialization
   - Can match dense network performance
   - "Winning tickets" through iterative pruning

Pruning Schedules:
- One-shot: Prune once, fine-tune
- Iterative: Gradual pruning with retraining
- Gradual magnitude pruning: Cubic sparsity schedule
  s_t = s_f + (s_i - s_f)(1 - (t-t_0)/(n∆t))^3

Importance Metrics:
• Weight magnitude: |w_i|
• Gradient-based: |w_i · ∂L/∂w_i|
• Fisher Information: (∂L/∂w_i)^2
• Taylor expansion: ∆L ≈ (∂L/∂w_i)·w_i

Advanced Methods:
- Movement pruning
- Variational dropout
- Soft pruning with learned gates
"""
            },
            "distillation": {
                "title": "Knowledge Distillation",
                "content": """
Knowledge Distillation: Teacher-Student Learning

Core Concept:
Transfer knowledge from large "teacher" model to compact "student" model

Distillation Loss:
L = α·L_CE(y, y_student) + (1-α)·L_KD(z_teacher, z_student)

Where:
- L_CE: Cross-entropy with true labels
- L_KD: Kullback-Leibler divergence
- z: Softened logits with temperature T

Temperature Scaling:
p_i = exp(z_i/T) / Σ_j exp(z_j/T)

Higher T → softer probability distributions
Reveals inter-class similarities

Distillation Variants:
1. Response-based:
   - Match final layer outputs
   - Original distillation approach
   
2. Feature-based:
   - Match intermediate representations
   - FitNets, Attention Transfer
   
3. Relation-based:
   - Match relationship between samples
   - RKD, Correlation Congruence

Advanced Techniques:
• Self-distillation: Teacher = Student architecture
• Multi-teacher distillation
• Online distillation: Co-training
• Cross-modal distillation
• Data-free distillation: Synthesized data
"""
            }
        }
    
    def _load_application_knowledge(self):
        """Load AI applications knowledge."""
        return {
            "nlp": {
                "title": "Natural Language Processing Applications",
                "content": """
Natural Language Processing: Language Understanding & Generation

Core NLP Tasks:
1. Language Modeling:
   - P(w_t | w_1, ..., w_{t-1})
   - Foundation for generation tasks
   - Perplexity metric: exp(-1/N Σ log P(w_i))
   
2. Machine Translation:
   - Seq2seq with attention
   - Transformer dominance (BLEU 40+)
   - Zero-shot translation via multilingual models
   
3. Question Answering:
   - Extractive: Span prediction from context
   - Abstractive: Generate novel answers
   - Open-domain: Retrieval + generation

State-of-the-Art Models:
- BERT: Masked language modeling
  • [MASK] token prediction
  • Next sentence prediction
  
- GPT Series: Autoregressive generation
  • GPT-3: 175B parameters
  • GPT-4: Multimodal capabilities
  
- T5: Unified text-to-text framework
  • All tasks as text generation
  • Flexible prompt engineering

Advanced Techniques:
• Prompt Engineering: In-context learning
• Few-shot Learning: Task adaptation with examples
• Chain-of-Thought: Reasoning step decomposition
• Retrieval-Augmented Generation (RAG)
• Fine-tuning strategies: LoRA, Adapter modules

Evaluation Metrics:
- BLEU, ROUGE: N-gram overlap
- BERTScore: Semantic similarity
- Human evaluation: Fluency, coherence
"""
            },
            "computer_vision": {
                "title": "Computer Vision Applications",
                "content": """
Computer Vision: Visual Understanding & Generation

Core CV Tasks:
1. Image Classification:
   - CNNs: Feature hierarchy extraction
   - ResNet: Skip connections, 1000+ layers
   - EfficientNet: Compound scaling (width, depth, resolution)
   - ViT: Transformer-based, patch embeddings
   
2. Object Detection:
   - Two-stage: R-CNN, Fast/Faster R-CNN, Mask R-CNN
   - One-stage: YOLO, SSD, RetinaNet
   - Anchor-free: FCOS, CenterNet
   - Loss: Classification + Bounding box regression
   
3. Semantic Segmentation:
   - Pixel-wise classification
   - U-Net: Encoder-decoder with skip connections
   - DeepLab: Atrous convolutions, ASPP
   - Metrics: IoU, Dice coefficient
   
4. Instance Segmentation:
   - Object detection + segmentation masks
   - Mask R-CNN: RoI pooling + mask head
   - SOLO: Segmenting Objects by Locations

Advanced Applications:
- Image Generation:
  • GANs: Adversarial training
  • Diffusion Models: DALL-E 2, Stable Diffusion
  • Score matching, denoising process
  
- 3D Vision:
  • NeRF: Neural Radiance Fields
  • 3D reconstruction, novel view synthesis
  • Volumetric rendering
  
- Video Understanding:
  • Action recognition: I3D, SlowFast
  • Temporal modeling: 3D CNNs, Video Transformers
  • Optical flow, motion prediction

Emerging Directions:
• Vision-Language Models: CLIP, ALIGN
• Self-supervised learning: SimCLR, MoCo
• Foundation models: SAM (Segment Anything)
"""
            },
            "reinforcement_learning": {
                "title": "Reinforcement Learning Applications",
                "content": """
Reinforcement Learning: Sequential Decision Making

Core RL Concepts:
1. Markov Decision Process (MDP):
   - States (S), Actions (A), Rewards (R)
   - Transition: P(s'|s,a)
   - Policy: π(a|s)
   - Value function: V^π(s) = E[Σ γ^t r_t | s_0=s]
   
2. Q-Learning:
   - Q(s,a): Action-value function
   - Bellman equation: Q(s,a) = r + γ max_a' Q(s',a')
   - Off-policy TD learning
   
3. Policy Gradient:
   - Direct policy optimization
   - ∇J(θ) = E[∇log π_θ(a|s) Q(s,a)]
   - REINFORCE algorithm

Deep RL Algorithms:
1. Value-Based:
   - DQN: Deep Q-Networks
     • Experience replay
     • Target network stabilization
   - Double DQN: Overestimation mitigation
   - Dueling DQN: Separate V(s) and A(s,a)
   
2. Policy-Based:
   - A3C: Asynchronous Advantage Actor-Critic
   - PPO: Proximal Policy Optimization
     • Clipped surrogate objective
     • Stable, sample-efficient
   - TRPO: Trust Region Policy Optimization
   
3. Model-Based:
   - World models: Learn environment dynamics
   - Planning in learned latent space
   - Dyna-Q: Integrated learning and planning

Applications:
• Game Playing:
  - AlphaGo: Monte Carlo Tree Search + Deep NN
  - Dota 2, StarCraft: Multi-agent learning
  
• Robotics:
  - Manipulation: Grasping, assembly
  - Locomotion: Bipedal walking, quadruped
  - Sim-to-real transfer: Domain randomization
  
• Autonomous Systems:
  - Self-driving: Path planning, control
  - Drone navigation: Visual servoing
  
• Resource Management:
  - Data center cooling
  - Traffic signal control
  - Portfolio optimization

Advanced Topics:
• Multi-agent RL: Nash equilibria
• Hierarchical RL: Temporal abstractions
• Inverse RL: Learning from demonstrations
• Meta-RL: Learning to learn quickly
• Offline RL: Learning from fixed datasets
"""
            }
        }
    
    def detect_trigger(self, user_input):
        """
        Detect if the trigger phrase is present in user input.
        
        Args:
            user_input (str): User's input text
            
        Returns:
            bool: True if trigger phrase detected
        """
        return self.trigger_phrase.lower() in user_input.lower()
    
    def get_random_technical_info(self):
        """
        Get random technical information from knowledge base.
        
        Returns:
            str: Formatted technical information
        """
        # Combine all topics
        all_topics = []
        all_topics.extend(list(self.deep_learning_topics.items()))
        all_topics.extend(list(self.optimization_topics.items()))
        all_topics.extend(list(self.application_topics.items()))
        
        # Select random topic
        topic_key, topic_data = random.choice(all_topics)
        
        response = f"\n{'='*70}\n"
        response += f"{topic_data['title']}\n"
        response += f"{'='*70}\n"
        response += topic_data['content']
        response += f"\n{'='*70}\n"
        
        return response
    
    def get_topic_by_keyword(self, keyword):
        """
        Get technical information based on keyword.
        
        Args:
            keyword (str): Search keyword
            
        Returns:
            str: Formatted technical information or None
        """
        keyword_lower = keyword.lower()
        
        # Search in all knowledge bases
        all_knowledge = {
            **self.deep_learning_topics,
            **self.optimization_topics,
            **self.application_topics
        }
        
        # First priority: exact match in topic key
        if keyword_lower in all_knowledge:
            topic_data = all_knowledge[keyword_lower]
            response = f"\n{'='*70}\n"
            response += f"{topic_data['title']}\n"
            response += f"{'='*70}\n"
            response += topic_data['content']
            response += f"\n{'='*70}\n"
            return response
        
        # Second priority: match in topic key or title
        for topic_key, topic_data in all_knowledge.items():
            if (keyword_lower in topic_key.lower() or 
                keyword_lower in topic_data['title'].lower()):
                
                response = f"\n{'='*70}\n"
                response += f"{topic_data['title']}\n"
                response += f"{'='*70}\n"
                response += topic_data['content']
                response += f"\n{'='*70}\n"
                
                return response
        
        # Third priority: match in content
        for topic_key, topic_data in all_knowledge.items():
            if keyword_lower in topic_data['content'].lower():
                
                response = f"\n{'='*70}\n"
                response += f"{topic_data['title']}\n"
                response += f"{'='*70}\n"
                response += topic_data['content']
                response += f"\n{'='*70}\n"
                
                return response
        
        return None
    
    def list_available_topics(self):
        """
        List all available topics.
        
        Returns:
            str: Formatted list of topics
        """
        response = "\n" + "="*70 + "\n"
        response += "Available Technical Discussion Topics\n"
        response += "="*70 + "\n\n"
        
        response += "1. Deep Learning Architectures:\n"
        for key, topic in self.deep_learning_topics.items():
            response += f"   - {topic['title']}\n"
        
        response += "\n2. Model Optimization:\n"
        for key, topic in self.optimization_topics.items():
            response += f"   - {topic['title']}\n"
        
        response += "\n3. AI Applications:\n"
        for key, topic in self.application_topics.items():
            response += f"   - {topic['title']}\n"
        
        response += "\n" + "="*70 + "\n"
        response += "\nTo explore a topic, mention keywords like:\n"
        response += "- 'transformer', 'neural network', 'activation'\n"
        response += "- 'quantization', 'pruning', 'distillation'\n"
        response += "- 'NLP', 'computer vision', 'reinforcement learning'\n"
        response += "\nOr say 'My Balorg is AI!' for a random technical deep dive!\n"
        response += "="*70 + "\n"
        
        return response
    
    def process_input(self, user_input):
        """
        Process user input and generate appropriate response.
        
        Args:
            user_input (str): User's input text
            
        Returns:
            str: AI response
        """
        if not user_input or not user_input.strip():
            return "Please provide input for technical discussion."
        
        # Check for trigger phrase
        if self.detect_trigger(user_input):
            return self.get_random_technical_info()
        
        # Check for help/list request
        user_input_lower = user_input.lower()
        if (user_input_lower in ['help', 'topics', 'list'] or 
            user_input_lower.startswith('show me topics') or
            user_input_lower.startswith('list topics')):
            return self.list_available_topics()
        
        # Try to find relevant topic by keyword
        keywords = ['transformer', 'neural', 'activation', 'quantization', 
                   'pruning', 'distillation', 'nlp', 'vision', 'reinforcement',
                   'bert', 'gpt', 'cnn', 'rnn', 'attention', 'optimization']
        
        for keyword in keywords:
            if keyword in user_input.lower():
                topic_info = self.get_topic_by_keyword(keyword)
                if topic_info:
                    return topic_info
        
        # Default response
        return (
            "\nI'm ready for technical discussions! You can:\n"
            "1. Say 'My Balorg is AI!' for a technical deep dive\n"
            "2. Ask about specific topics (transformers, quantization, NLP, etc.)\n"
            "3. Type 'help' or 'topics' to see all available topics\n"
        )
    
    def run_interactive(self):
        """Run interactive CLI mode."""
        print("\n" + "="*70)
        print("Balorg AI - Advanced Technical Discussion System")
        print("="*70)
        print("\nWelcome! I'm an AI assistant specializing in:")
        print("• Deep Learning Architectures")
        print("• Model Optimization Techniques")
        print("• AI Applications (NLP, Computer Vision, RL)")
        print("\nSay 'My Balorg is AI!' to trigger technical information!")
        print("Type 'help' to see all topics, or 'exit' to quit.")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nThank you for the technical discussion! Goodbye!")
                    break
                
                response = self.process_input(user_input)
                print(response)
                
            except (EOFError, KeyboardInterrupt):
                print("\n\nThank you for the technical discussion! Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")


def main():
    """Main entry point for the Balorg AI system."""
    balorg = BalorgAI()
    
    # Check if running in interactive mode or with arguments
    if len(sys.argv) > 1:
        # Process command line input
        user_input = " ".join(sys.argv[1:])
        response = balorg.process_input(user_input)
        print(response)
    else:
        # Run interactive mode
        balorg.run_interactive()


if __name__ == "__main__":
    main()
