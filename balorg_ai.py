#!/usr/bin/env python3
"""
Balorg AI - A Technical Discussion Facilitator
Responds to "My Balorg is AI!" with highly technical information about AI topics.
"""

import sys
import random


class BalorgAI:
    """Main class for the Balorg AI technical discussion system."""
    
    def __init__(self):
        self.trigger_phrase = "My Balorg is AI!"
        self.knowledge_base = self._initialize_knowledge_base()
        # Create reverse lookup for category titles to keys for O(1) lookup
        self.title_to_key = {
            value['title'].lower(): key 
            for key, value in self.knowledge_base.items()
        }
    
    def _initialize_knowledge_base(self):
        """Initialize the technical knowledge base."""
        return {
            "deep_learning_architectures": {
                "title": "Deep Learning Architectures",
                "topics": [
                    {
                        "name": "Transformer Models",
                        "content": """
Transformer Architecture (Vaswani et al., 2017):
- Self-attention mechanism with Query, Key, Value matrices
- Multi-head attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Positional encoding: PE(pos,2i) = sin(pos/10000^(2i/d_model))
- Layer normalization and residual connections
- Encoder-decoder structure with masked self-attention in decoder

Key Innovations:
- Eliminates recurrence, enabling parallel processing
- O(1) sequential operations vs O(n) in RNNs
- Attention weights provide interpretability
- Scales to massive datasets (GPT-3: 175B parameters)

Variants:
- BERT: Bidirectional encoder with masked language modeling
- GPT: Unidirectional decoder with causal masking
- Vision Transformer (ViT): 16x16 patches with position embeddings
- Sparse Transformers: O(n√n) complexity with factorized attention
"""
                    },
                    {
                        "name": "Neural Network Architectures",
                        "content": """
Advanced Neural Network Architectures:

1. Convolutional Neural Networks (CNNs):
   - Convolution: (f * g)(t) = ∫ f(τ)g(t-τ)dτ
   - Receptive field growth: r_l = r_(l-1) + (k-1) * ∏(s_i)
   - ResNet: Skip connections solve vanishing gradients
   - DenseNet: Dense connectivity with feature reuse
   - EfficientNet: Compound scaling (depth, width, resolution)

2. Recurrent Architectures:
   - LSTM gates: f_t = σ(W_f·[h_(t-1),x_t] + b_f)
   - GRU: Simplified gating with reset and update gates
   - Bidirectional processing for context awareness
   
3. Graph Neural Networks (GNNs):
   - Message passing: h_v^(k) = UPDATE(h_v^(k-1), AGG({h_u^(k-1) : u ∈ N(v)}))
   - Graph convolution: H^(l+1) = σ(D^(-1/2)AD^(-1/2)H^(l)W^(l))
   - Applications: molecular property prediction, social networks

4. Neural Architecture Search (NAS):
   - Automated architecture optimization
   - DARTS: Differentiable architecture search
   - Reduces design time from months to days
"""
                    },
                    {
                        "name": "Attention Mechanisms",
                        "content": """
Attention Mechanism Deep Dive:

Scaled Dot-Product Attention:
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- Scaling factor √d_k prevents vanishing gradients
- Complexity: O(n²d) where n is sequence length

Multi-Head Attention:
- MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
- head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
- Allows model to attend to different representation subspaces
- Typical: 8-16 heads with d_k = d_model/h

Attention Variants:
- Cross-attention: Q from one sequence, K,V from another
- Self-attention: Q,K,V all from same sequence
- Sparse attention: Reduces complexity to O(n√n) or O(n log n)
- Linear attention: Kernel-based reformulation O(nd²)
- Flash Attention: Memory-efficient GPU implementation

Applications:
- Sequence-to-sequence tasks (translation, summarization)
- Vision: Spatial attention in image regions
- Multi-modal: Cross-modal attention (text-image alignment)
"""
                    }
                ]
            },
            "model_optimization": {
                "title": "Model Optimization Techniques",
                "topics": [
                    {
                        "name": "Quantization",
                        "content": """
Neural Network Quantization:

Post-Training Quantization (PTQ):
- INT8 quantization: x_q = round(x/s) + z
  where s = (x_max - x_min)/(2^b - 1)
- Symmetric: z = 0, range = [-127, 127]
- Asymmetric: Full range utilization
- Achieves 4x memory reduction, 2-4x speedup

Quantization-Aware Training (QAT):
- Simulates quantization during training
- Straight-through estimator for gradients
- Better accuracy than PTQ (≤1% degradation)
- Fake quantization: x_fake = (round(x/s) * s)

Advanced Techniques:
- Mixed precision: Critical layers in FP16/FP32
- Per-channel quantization: Separate scales per output channel
- Dynamic quantization: Runtime weight quantization
- GPTQ: Optimal Brain Quantization for LLMs

Bit-width Options:
- INT8: Standard, 4x compression
- INT4: 8x compression, LLM deployment
- Binary/Ternary: Extreme compression, accuracy loss
- BFloat16: Training-friendly 16-bit format

Hardware Support:
- NVIDIA Tensor Cores: INT8/INT4 operations
- Intel VNNI: AVX-512 INT8 acceleration
- ARM: NEON SIMD instructions
"""
                    },
                    {
                        "name": "Pruning",
                        "content": """
Neural Network Pruning:

Magnitude-Based Pruning:
- Remove weights where |W_ij| < threshold
- Iterative pruning: Train → Prune → Fine-tune
- Can achieve 90% sparsity with minimal accuracy loss
- Lottery Ticket Hypothesis: Sparse subnetworks exist from initialization

Structured Pruning:
- Channel pruning: Remove entire filters
- Layer pruning: Remove entire layers
- Hardware-friendly: Actual speedup on standard hardware
- Filter importance: L1-norm, Taylor expansion, gradient-based

Gradient-Based Pruning:
- Optimal Brain Damage: Second-order Taylor approximation
- Importance score: I_i = (∂L/∂w_i)² * w_i²/2
- Fisher information for parameter importance

Dynamic Pruning:
- Runtime adaptive sparsity
- Conditional computation: Skip unnecessary paths
- Early exit strategies for inference

Pruning Schedules:
- One-shot: Single pruning step
- Iterative: Gradual sparsity increase
- Cubic schedule: s_t = s_f + (s_i - s_f)(1 - (t-t_0)/(n∆t))³
  
Benefits:
- 10-100x compression possible
- Reduced memory footprint
- Lower inference latency
- Energy efficiency improvements
"""
                    },
                    {
                        "name": "Knowledge Distillation",
                        "content": """
Knowledge Distillation Framework:

Core Concept (Hinton et al., 2015):
- Teacher model (large) → Student model (small)
- Soft targets contain dark knowledge
- Temperature scaling: p_i = exp(z_i/T) / Σ exp(z_j/T)
- Loss: L = αL_CE(y, ŷ_student) + (1-α)L_KL(ŷ_teacher, ŷ_student)

Distillation Variants:

1. Response-Based:
   - Match output probabilities
   - Soft labels more informative than hard labels
   - Temperature T typically 2-10

2. Feature-Based:
   - Match intermediate representations
   - FitNet: Thin deep networks via hint layers
   - Attention transfer: Match attention maps

3. Relation-Based:
   - FSP matrix: Gram matrix of feature correlations
   - Similarity-preserving distillation

Advanced Techniques:
- Self-distillation: Model teaches itself
- Multi-teacher distillation: Ensemble knowledge
- Cross-architecture: CNN → Transformer
- Online distillation: Simultaneous training

Applications:
- BERT → DistilBERT: 40% size, 97% performance
- Large LLMs → Smaller deployment models
- Edge device deployment
- Real-time inference systems

Theoretical Understanding:
- Reduces model variance
- Implicit regularization effect
- Label smoothing connection
- Compression-generalization tradeoff
"""
                    }
                ]
            },
            "ai_applications": {
                "title": "AI Applications",
                "topics": [
                    {
                        "name": "Natural Language Processing",
                        "content": """
Modern Natural Language Processing:

Language Models:
- Autoregressive: P(x) = ∏ P(x_i|x_<i)
- BERT: Masked language modeling with bidirectional context
- GPT architecture: Causal attention for generation
- T5: Text-to-text unified framework
- Parameter scaling laws: Performance ∝ N^α (α ≈ 0.076)

Pre-training Objectives:
- MLM: Masked language modeling
- NSP: Next sentence prediction
- RTD: Replaced token detection (ELECTRA)
- ELECTRA: 30x more efficient than BERT

Fine-tuning Strategies:
- Full fine-tuning: All parameters updated
- Adapter layers: Insert trainable modules
- LoRA: Low-rank adaptation W = W_0 + BA (B,A low-rank)
- Prefix tuning: Learn task-specific prefixes
- P-tuning: Prompt-based parameter-efficient tuning

NLP Tasks:
- Sequence classification: Sentiment, intent detection
- Token classification: NER, POS tagging
- Question answering: Extractive, generative
- Summarization: Abstractive, extractive
- Translation: Seq2seq with attention

Evaluation Metrics:
- BLEU: n-gram precision (translation)
- ROUGE: Recall-oriented summarization
- BERTScore: Embedding-based similarity
- Perplexity: exp(−1/N Σ log P(x_i|x_<i))

State-of-the-Art Models:
- GPT-4: Multi-modal reasoning
- Claude: Constitutional AI alignment
- PaLM: Pathways language model (540B)
- LLaMA: Open-source efficient models
"""
                    },
                    {
                        "name": "Computer Vision",
                        "content": """
Advanced Computer Vision:

Image Classification:
- CNNs: Hierarchical feature learning
- Vision Transformers: Patch-based self-attention
- Hybrid models: CoAtNet, ConvNeXt
- Few-shot learning: Prototypical networks, MAML

Object Detection:
- Two-stage: R-CNN family (Faster R-CNN)
  * RPN: Region proposal network
  * RoI pooling for feature extraction
- One-stage: YOLO, SSD, RetinaNet
  * Focal loss: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
- Anchor-free: FCOS, CenterNet
- Transformer-based: DETR (end-to-end detection)

Segmentation:
- Semantic: FCN, DeepLab, PSPNet
  * Atrous convolution for large receptive fields
- Instance: Mask R-CNN (detection + segmentation)
- Panoptic: Unified thing + stuff segmentation

Self-Supervised Learning:
- Contrastive: SimCLR, MoCo
  * InfoNCE loss: L = -log[exp(z_i·z_j/τ)/Σ exp(z_i·z_k/τ)]
- Non-contrastive: BYOL, SimSiam
- Masked image modeling: MAE, BEiT

3D Vision:
- NeRF: Neural radiance fields
  * F(x,y,z,θ,φ) → (RGB, density)
- Point cloud processing: PointNet, PointNet++
- Multi-view geometry: Structure from motion

Applications:
- Medical imaging: Tumor detection, organ segmentation
- Autonomous driving: Scene understanding, path planning
- Augmented reality: SLAM, object tracking
- Industrial inspection: Defect detection
"""
                    },
                    {
                        "name": "Reinforcement Learning",
                        "content": """
Reinforcement Learning Theory:

Markov Decision Process (MDP):
- State space S, action space A, transition P(s'|s,a)
- Reward function R(s,a,s')
- Policy π: S → Δ(A)
- Value function: V^π(s) = E[Σ γ^t r_t | s_0=s, π]
- Q-function: Q^π(s,a) = E[Σ γ^t r_t | s_0=s, a_0=a, π]

Value-Based Methods:
- Q-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
- DQN: Deep Q-Network with experience replay
- Double DQN: Reduces overestimation bias
- Dueling DQN: V(s) + A(s,a) - mean(A(s,·))
- Rainbow: Combines 6 DQN improvements

Policy Gradient Methods:
- REINFORCE: ∇_θ J(θ) = E[∇_θ log π_θ(a|s) Q^π(s,a)]
- Actor-Critic: Combines value and policy
- A3C: Asynchronous advantage actor-critic
- PPO: Clipped surrogate objective
  * L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ),1-ε,1+ε)Â_t)]
- TRPO: Trust region optimization

Model-Based RL:
- World models: Learn environment dynamics
- Dyna-Q: Model learning + planning
- MuZero: Learned model for planning (AlphaGo successor)
- Dreamer: Latent world models

Advanced Topics:
- Multi-agent RL: Nash equilibria, communication
- Hierarchical RL: Temporal abstraction, options
- Inverse RL: Learn reward from demonstrations
- Meta-RL: Learning to learn across tasks

Applications:
- Game playing: AlphaGo, OpenAI Five
- Robotics: Manipulation, locomotion
- Recommendation systems: Personalization
- Resource management: Data centers, traffic control
"""
                    }
                ]
            }
        }
    
    def detect_trigger(self, text):
        """Check if the trigger phrase is present in the text."""
        return self.trigger_phrase.lower() in text.lower()
    
    def get_technical_response(self, category=None):
        """Generate a technical response based on the category."""
        if category is None:
            # Select a random category
            category = random.choice(list(self.knowledge_base.keys()))
        
        if category not in self.knowledge_base:
            return "I don't have information on that topic yet."
        
        topic_data = self.knowledge_base[category]
        selected_topic = random.choice(topic_data["topics"])
        
        response = f"\n{'='*80}\n"
        response += f"BALORG AI TECHNICAL RESPONSE\n"
        response += f"Category: {topic_data['title']}\n"
        response += f"Topic: {selected_topic['name']}\n"
        response += f"{'='*80}\n"
        response += selected_topic['content']
        response += f"\n{'='*80}\n"
        
        return response
    
    def list_categories(self):
        """List all available technical categories."""
        print("\nAvailable technical discussion topics:")
        for idx, (key, value) in enumerate(self.knowledge_base.items(), 1):
            print(f"{idx}. {value['title']}")
            for topic in value['topics']:
                print(f"   - {topic['name']}")
        print()
    
    def interactive_mode(self):
        """Run the interactive conversation mode."""
        print("\n" + "="*80)
        print("BALORG AI - TECHNICAL DISCUSSION FACILITATOR")
        print("="*80)
        print("\nWelcome! I'm here to discuss advanced AI topics.")
        print(f"Say '{self.trigger_phrase}' to receive technical information.")
        print("Type 'topics' to see available categories.")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit']:
                    print("\nThank you for the technical discussion! Goodbye.\n")
                    break
                
                if user_input.lower() == 'topics':
                    self.list_categories()
                    continue
                
                if self.detect_trigger(user_input):
                    # Check if user specified a category using optimized lookup
                    category = None
                    user_input_lower = user_input.lower()
                    for title_lower, key in self.title_to_key.items():
                        if title_lower in user_input_lower:
                            category = key
                            break
                    
                    response = self.get_technical_response(category)
                    print(f"\nBalorg: {response}")
                else:
                    print("\nBalorg: To access my technical knowledge, please say")
                    print(f"        '{self.trigger_phrase}'")
                    print("        You can also specify a topic like:")
                    print(f"        '{self.trigger_phrase}' (Deep Learning)")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!\n")
                break
            except EOFError:
                print("\n\nGoodbye!\n")
                break


def main():
    """Main entry point for the Balorg AI application."""
    balorg = BalorgAI()
    
    # Check if running in interactive mode or with arguments
    if len(sys.argv) > 1:
        # Command-line mode
        user_input = ' '.join(sys.argv[1:])
        
        if user_input.lower() == 'topics':
            balorg.list_categories()
        elif balorg.detect_trigger(user_input):
            print(balorg.get_technical_response())
        else:
            print(f"To trigger technical response, use: '{balorg.trigger_phrase}'")
    else:
        # Interactive mode
        balorg.interactive_mode()


if __name__ == "__main__":
    main()
