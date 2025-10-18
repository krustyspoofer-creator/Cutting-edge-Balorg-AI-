#!/usr/bin/env python3
"""
Cutting-edge Balorg AI - Interactive Technical Discussion System

This system facilitates technical discussions about AI topics.
Say "My Balorg is AI!" to trigger highly technical responses.
"""

import sys
import random


class BalorgAI:
    """
    An interactive AI discussion system that provides technical information
    when triggered by the phrase "My Balorg is AI!"
    """
    
    def __init__(self):
        self.technical_mode = False
        self.topics = {
            'deep_learning': self._get_deep_learning_topics(),
            'optimization': self._get_optimization_topics(),
            'applications': self._get_application_topics()
        }
        
    def _get_deep_learning_topics(self):
        """Technical information about deep learning architectures"""
        return [
            {
                'title': 'Transformer Architecture',
                'content': """
**Transformer Architecture - Technical Deep Dive**

Core Components:
1. Multi-Head Self-Attention Mechanism:
   - Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
   - Enables parallel processing of sequence data
   - Computational complexity: O(n²·d) where n is sequence length
   
2. Positional Encoding:
   - PE(pos,2i) = sin(pos/10000^(2i/d_model))
   - PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
   - Injects sequence order information without recurrence
   
3. Layer Normalization:
   - Applied before each sub-layer (Pre-LN) or after (Post-LN)
   - Stabilizes training dynamics in deep networks
   
4. Feed-Forward Networks:
   - FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
   - Typical dimension expansion: d_model → 4·d_model → d_model

Architectural Variants:
- BERT: Bidirectional encoder-only, masked language modeling
- GPT: Unidirectional decoder-only, autoregressive generation
- T5: Encoder-decoder, text-to-text framework
- Vision Transformer (ViT): Applies transformers to image patches
"""
            },
            {
                'title': 'Neural Network Fundamentals',
                'content': """
**Advanced Neural Network Architectures**

Convolutional Neural Networks (CNNs):
1. Convolutional Layer Mathematics:
   - Output feature map: Y[i,j] = Σ(X[i+m,j+n] · K[m,n]) + b
   - Receptive field grows with depth: RF_l = RF_(l-1) + (k-1)·∏stride_i
   
2. Advanced CNN Architectures:
   - ResNet: Skip connections mitigate vanishing gradients
     - y = F(x, {W_i}) + x (identity mapping)
   - DenseNet: Each layer connected to all subsequent layers
     - Feature reuse, reduced parameters
   - EfficientNet: Compound scaling (depth, width, resolution)
     - Achieves SOTA with fewer parameters

Recurrent Neural Networks (RNNs):
1. LSTM Architecture:
   - Forget gate: f_t = σ(W_f·[h_(t-1), x_t] + b_f)
   - Input gate: i_t = σ(W_i·[h_(t-1), x_t] + b_i)
   - Cell state: C_t = f_t ⊙ C_(t-1) + i_t ⊙ tanh(W_C·[h_(t-1), x_t] + b_C)
   - Output gate: o_t = σ(W_o·[h_(t-1), x_t] + b_o)
   - Hidden state: h_t = o_t ⊙ tanh(C_t)
   
2. GRU (Gated Recurrent Unit):
   - Simplified architecture with fewer gates
   - Often performs comparably to LSTM with less computation
"""
            },
            {
                'title': 'Attention Mechanisms',
                'content': """
**Attention Mechanisms - Mathematical Framework**

Self-Attention Formulation:
1. Query-Key-Value Paradigm:
   - Q = XW_Q, K = XW_K, V = XW_V
   - Attention weights: α = softmax(QK^T/√d_k)
   - Output: Z = αV
   
2. Multi-Head Attention:
   - head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
   - MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
   - Benefits: Captures different representation subspaces
   
3. Cross-Attention:
   - Q from decoder, K and V from encoder
   - Enables information flow between sequences
   
Advanced Attention Variants:
- Linear Attention: O(n) complexity via kernel approximation
- Sparse Attention: Attends to subset of positions
- Flash Attention: IO-aware algorithm for memory efficiency
- Relative Position Attention: Explicit position relationships

Attention Optimization:
- Gradient checkpointing to reduce memory
- Mixed precision training (FP16/BF16)
- Attention masking for causal/padding positions
"""
            }
        ]
    
    def _get_optimization_topics(self):
        """Technical information about model optimization"""
        return [
            {
                'title': 'Quantization Techniques',
                'content': """
**Model Quantization - Precision Reduction Strategies**

Quantization Fundamentals:
1. Uniform Quantization:
   - Q(x) = round((x - z)/s)
   - Dequantization: x̃ = s·Q(x) + z
   - where s = scale, z = zero-point
   
2. Quantization Schemes:
   - Symmetric: z = 0, range = [-max, max]
   - Asymmetric: z ≠ 0, range = [min, max]
   - Per-tensor vs per-channel granularity

Advanced Techniques:
1. Post-Training Quantization (PTQ):
   - Dynamic quantization: activations quantized at runtime
   - Static quantization: calibration on representative dataset
   - Weight-only quantization for memory reduction
   
2. Quantization-Aware Training (QAT):
   - Simulates quantization during training
   - Straight-through estimator for gradient flow
   - Typically achieves better accuracy than PTQ
   
3. Mixed Precision Quantization:
   - Different layers use different bit-widths
   - Sensitive layers (e.g., first/last) use higher precision
   - AutoML approaches for bit-width selection

Hardware Considerations:
- INT8: 4x memory reduction, 2-4x speedup on modern CPUs
- INT4/INT2: Extreme compression for edge devices
- BFloat16: Good balance of range and precision
"""
            },
            {
                'title': 'Neural Network Pruning',
                'content': """
**Pruning Strategies for Model Compression**

Pruning Categories:
1. Structured Pruning:
   - Removes entire neurons, channels, or layers
   - Maintains regular structure for hardware efficiency
   - Typically 2-3x speedup with minimal accuracy loss
   
2. Unstructured Pruning:
   - Removes individual weights (creates sparse tensors)
   - Higher compression ratios possible
   - Requires specialized sparse kernels for speedup

Pruning Criteria:
1. Magnitude-based Pruning:
   - Remove weights with smallest |W_ij|
   - Simple but effective baseline
   
2. Gradient-based Importance:
   - Taylor expansion: ΔL ≈ |g_i · w_i|
   - Prune parameters with minimal impact on loss
   
3. Second-order Methods:
   - Optimal Brain Damage: Uses Hessian diagonal
   - Optimal Brain Surgeon: Full Hessian information

Pruning Schedules:
1. One-shot Pruning:
   - Prune once after training
   - Fast but potentially suboptimal
   
2. Iterative Magnitude Pruning (IMP):
   - Prune → Fine-tune → Repeat
   - Lottery Ticket Hypothesis: Finds winning subnetworks
   
3. Gradual Pruning:
   - Sparsity increases during training
   - s_t = s_f + (s_i - s_f)(1 - (t/T)³)
"""
            },
            {
                'title': 'Knowledge Distillation',
                'content': """
**Knowledge Distillation - Model Compression via Teacher-Student**

Core Methodology:
1. Distillation Loss:
   - L_KD = (1-α)L_CE(y, ŷ_s) + α·T²·L_KL(σ(z_t/T), σ(z_s/T))
   - α: distillation weight, T: temperature
   - Soft targets preserve dark knowledge
   
2. Temperature Scaling:
   - High T: Smoother probability distribution
   - Reveals similarity structure in teacher's outputs
   - Typical values: T ∈ [2, 20]

Advanced Distillation Techniques:
1. Feature-based Distillation:
   - Match intermediate layer representations
   - L_feat = ||h_s - W·h_t||²
   - Captures richer knowledge than logits alone
   
2. Attention Transfer:
   - Distill attention maps between teacher/student
   - Guides student to focus on same regions
   
3. Self-Distillation:
   - Model is its own teacher (from earlier checkpoint)
   - Born-Again Networks: Iterative refinement

Applications:
- Large-to-small model compression (BERT → DistilBERT)
- Ensemble distillation (many teachers → one student)
- Cross-modal distillation (different modalities)
- Continual learning (prevent catastrophic forgetting)

Optimization Considerations:
- Teacher freeze vs. co-training
- Layer matching strategies
- Balancing multiple distillation objectives
"""
            }
        ]
    
    def _get_application_topics(self):
        """Technical information about AI applications"""
        return [
            {
                'title': 'Natural Language Processing',
                'content': """
**NLP State-of-the-Art Techniques**

Large Language Models (LLMs):
1. Architecture Components:
   - Transformer-based decoder stack
   - Typical sizes: 7B-175B parameters
   - Context windows: 2K-100K+ tokens
   
2. Training Methodology:
   - Self-supervised learning on massive text corpora
   - Next-token prediction objective
   - Compute-optimal scaling (Chinchilla laws):
     * Parameters and tokens should scale proportionally
     * N_tokens ≈ 20 × N_params for optimal training
   
3. Fine-tuning Strategies:
   - Instruction tuning: Task-specific prompts
   - RLHF (Reinforcement Learning from Human Feedback):
     * Reward model trained on human preferences
     * PPO optimization: max E[r(x,y) - β·KL(π||π_ref)]
   - LoRA: Low-Rank Adaptation for efficient fine-tuning
     * ΔW = BA where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

Advanced NLP Tasks:
1. Named Entity Recognition (NER):
   - Sequence labeling with BIO/BILOU tagging
   - CRF layer for structured prediction
   
2. Semantic Parsing:
   - Text → Logical form/SQL/Code
   - Seq2seq with copy mechanism
   
3. Coreference Resolution:
   - Span representations and pairwise scoring
   - Clustering algorithms for entity linking

Evaluation Metrics:
- BLEU, ROUGE, METEOR for generation
- F1, Exact Match for QA systems
- Perplexity for language modeling
"""
            },
            {
                'title': 'Computer Vision',
                'content': """
**Computer Vision - Deep Learning Approaches**

Object Detection:
1. Two-Stage Detectors:
   - R-CNN family: Region proposal → CNN classification
   - Faster R-CNN: Region Proposal Network (RPN)
     * Anchor boxes at multiple scales/ratios
     * Non-Maximum Suppression (NMS) for duplicate removal
   
2. One-Stage Detectors:
   - YOLO: Direct prediction on grid cells
     * Real-time processing: 30-60 FPS
   - RetinaNet: Focal Loss for class imbalance
     * FL(p_t) = -(1-p_t)^γ log(p_t)
   
3. Transformer-based Detection:
   - DETR: Direct set prediction with transformers
   - No anchor boxes or NMS required
   - Bipartite matching loss with Hungarian algorithm

Semantic Segmentation:
1. Fully Convolutional Networks:
   - U-Net architecture: Encoder-decoder with skip connections
   - Achieves pixel-level predictions
   
2. Advanced Architectures:
   - DeepLab: Atrous convolution for multi-scale features
   - Mask R-CNN: Instance segmentation extension
   - Segment Anything Model (SAM): Promptable segmentation

Vision-Language Models:
1. CLIP (Contrastive Language-Image Pre-training):
   - Joint embedding space for text and images
   - Contrastive loss: L = -log(exp(sim(I,T)/τ)/Σexp(sim(I,T_i)/τ))
   - Zero-shot transfer to downstream tasks
   
2. Applications:
   - Image captioning with attention
   - Visual question answering (VQA)
   - Text-to-image generation (DALL-E, Stable Diffusion)
"""
            },
            {
                'title': 'Reinforcement Learning',
                'content': """
**Reinforcement Learning - Advanced Algorithms**

Core Framework:
1. Markov Decision Process (MDP):
   - States S, Actions A, Rewards R, Transitions P
   - Goal: Maximize expected return G_t = Σ(γ^k · r_(t+k+1))
   
2. Value Functions:
   - State-value: V^π(s) = E_π[G_t | S_t = s]
   - Action-value: Q^π(s,a) = E_π[G_t | S_t = s, A_t = a]
   - Bellman equation: V(s) = Σ_a π(a|s)Σ_(s',r) p(s',r|s,a)[r + γV(s')]

Deep RL Algorithms:
1. Deep Q-Network (DQN):
   - Q-learning with neural network approximation
   - Experience replay for sample efficiency
   - Target network for stability: θ^- ← θ every C steps
   - Loss: L = E[(r + γ max_a' Q(s',a';θ^-) - Q(s,a;θ))²]
   
2. Policy Gradient Methods:
   - REINFORCE: ∇_θ J(θ) = E[∇_θ log π_θ(a|s)·G_t]
   - Actor-Critic: Combines value and policy learning
   - A3C: Asynchronous parallel training
   
3. Proximal Policy Optimization (PPO):
   - Trust region optimization for stable updates
   - Clipped objective: L^CLIP = E[min(r_t(θ)Â_t, clip(r_t(θ),1-ε,1+ε)Â_t)]
   - State-of-the-art for continuous control

Advanced Topics:
- Multi-agent RL: Nash equilibria, cooperative strategies
- Inverse RL: Learning reward functions from demonstrations
- Meta-RL: Learning to learn across task distributions
- Model-based RL: Learning world models for planning
  * Dyna-Q, MBPO, DreamerV2/V3

Applications:
- Game playing (AlphaGo, Dota 2, StarCraft II)
- Robotics control and manipulation
- Autonomous driving decision-making
- Resource allocation and optimization
"""
            }
        ]
    
    def process_input(self, user_input):
        """Process user input and return appropriate response"""
        normalized_input = user_input.strip()
        
        # Check for trigger phrase
        if "my balorg is ai" in normalized_input.lower():
            self.technical_mode = True
            return self._get_technical_response()
        
        # Check for exit commands
        if normalized_input.lower() in ['exit', 'quit', 'bye']:
            return None
        
        # Check for help command
        if normalized_input.lower() in ['help', '?']:
            return self._get_help_message()
        
        # Check for topic requests
        if 'deep learning' in normalized_input.lower() or 'transformer' in normalized_input.lower() or 'neural network' in normalized_input.lower():
            if self.technical_mode:
                return self._get_topic_detail('deep_learning')
            else:
                return self._get_layman_response('deep_learning')
        
        if 'optimization' in normalized_input.lower() or 'quantization' in normalized_input.lower() or 'pruning' in normalized_input.lower():
            if self.technical_mode:
                return self._get_topic_detail('optimization')
            else:
                return self._get_layman_response('optimization')
        
        if 'application' in normalized_input.lower() or 'nlp' in normalized_input.lower() or 'computer vision' in normalized_input.lower() or 'reinforcement' in normalized_input.lower():
            if self.technical_mode:
                return self._get_topic_detail('applications')
            else:
                return self._get_layman_response('applications')
        
        # Default conversational response
        return self._get_default_response()
    
    def _get_technical_response(self):
        """Get technical response when trigger phrase is detected"""
        response = """
╔══════════════════════════════════════════════════════════════════════════╗
║                  TECHNICAL MODE ACTIVATED                                 ║
║              Balorg AI - Advanced Technical Discussion                    ║
╚══════════════════════════════════════════════════════════════════════════╝

Welcome to the technical deep dive! I'm now providing highly specialized 
technical information. Available topics:

1. DEEP LEARNING ARCHITECTURES
   - Transformer models and attention mechanisms
   - CNN/RNN architectures and variants
   - Advanced neural network designs

2. MODEL OPTIMIZATION
   - Quantization techniques (INT8, mixed precision)
   - Neural network pruning strategies
   - Knowledge distillation methods

3. AI APPLICATIONS
   - Natural Language Processing (LLMs, NLP pipelines)
   - Computer Vision (object detection, segmentation)
   - Reinforcement Learning (DQN, PPO, policy gradients)

Commands:
- Ask about specific topics by name
- Type 'help' for more information
- Type 'exit' to quit

What technical topic would you like to explore?
"""
        return response
    
    def _get_topic_detail(self, topic_key):
        """Get detailed technical information for a topic"""
        topics = self.topics[topic_key]
        response = "\n" + "="*80 + "\n"
        for topic in topics:
            response += f"\n{topic['title']}\n"
            response += "="*80 + "\n"
            response += topic['content'] + "\n"
        return response
    
    def _get_layman_response(self, topic_key):
        """Get simplified response for layman terms"""
        layman_responses = {
            'deep_learning': """
Deep Learning is about teaching computers to learn from examples, much like 
how humans learn. It uses artificial neural networks - systems inspired by 
the human brain. These networks can:
- Recognize patterns in images (like identifying cats in photos)
- Understand and generate text (like chatbots)
- Make predictions based on past data

Say "My Balorg is AI!" if you want the full technical details with equations
and architectural specifications!
""",
            'optimization': """
Model Optimization is about making AI systems faster and smaller while keeping
them accurate. Think of it like compressing a large file - we want to reduce
the size without losing important information. Techniques include:
- Making numbers simpler (quantization)
- Removing unnecessary parts (pruning)
- Teaching smaller models from larger ones (distillation)

Say "My Balorg is AI!" for in-depth technical explanations with mathematical
formulations!
""",
            'applications': """
AI Applications are the real-world uses of artificial intelligence:
- Natural Language Processing: Teaching computers to understand human language
- Computer Vision: Helping computers "see" and understand images
- Reinforcement Learning: Teaching AI to make decisions through trial and error

Say "My Balorg is AI!" to dive deep into the technical implementations,
algorithms, and state-of-the-art methods!
"""
        }
        return layman_responses.get(topic_key, self._get_default_response())
    
    def _get_help_message(self):
        """Get help message"""
        return """
╔══════════════════════════════════════════════════════════════════════════╗
║                     Balorg AI Help System                                 ║
╚══════════════════════════════════════════════════════════════════════════╝

COMMANDS:
- 'My Balorg is AI!' - Activate technical mode for advanced discussions
- 'deep learning' - Learn about neural architectures
- 'optimization' - Learn about model compression
- 'applications' - Learn about AI use cases
- 'help' or '?' - Show this help message
- 'exit' or 'quit' - Exit the program

MODES:
- Layman Mode (default): Simplified explanations for general understanding
- Technical Mode: Activated by "My Balorg is AI!" - provides detailed
  mathematical formulations, algorithms, and architectural specifications

Ask any question about AI topics, and I'll respond accordingly!
"""
    
    def _get_default_response(self):
        """Get default conversational response"""
        responses = [
            "I can help you learn about AI! Ask me about deep learning, optimization, or applications.",
            "Interested in AI? I can explain topics in simple terms, or say 'My Balorg is AI!' for technical details.",
            "I'm here to discuss artificial intelligence. What would you like to know?",
            "Ask me about transformers, neural networks, or any AI topic. Say 'My Balorg is AI!' for advanced technical information."
        ]
        return random.choice(responses)
    
    def run(self):
        """Run the interactive chatbot"""
        print("\n" + "="*80)
        print("  BALORG AI - Cutting-edge Technical Discussion System")
        print("="*80)
        print("\nWelcome! I facilitate technical discussions about AI.")
        print("Say 'My Balorg is AI!' to activate technical mode.")
        print("Type 'help' for more information or 'exit' to quit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                
                response = self.process_input(user_input)
                
                if response is None:
                    print("\nBalorg AI: Thank you for the technical discussion! Goodbye.\n")
                    break
                
                print(f"\nBalorg AI: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nBalorg AI: Session interrupted. Goodbye!\n")
                break
            except EOFError:
                print("\n\nBalorg AI: End of input. Goodbye!\n")
                break


def main():
    """Main entry point"""
    ai = BalorgAI()
    ai.run()


if __name__ == "__main__":
    main()
