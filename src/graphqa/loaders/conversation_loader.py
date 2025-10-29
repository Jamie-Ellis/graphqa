"""
Conversation Analysis Loader for GraphQA

Loads ConvoKit corpus with Phase 2 metadata (message classification,
topic modeling, commitment detection) into NetworkX graph for relationship analysis.

Dataset: 31,249 multilingual messages (REALTALK English + WhatsApp Dutch)
Phase 2 Metadata:
  - Message Classification: questions/answers/statements (95%+ accuracy)
  - Topic Modeling: 119 topics (25 REALTALK + 94 WhatsApp), 80.6% topic shifts
    Uses separate BERTopic models per dataset with language-specific vectorizers
  - Commitment Detection: 20,983 commitments (67.1%), separate EN/NL models
    REALTALK: 91.5% detection rate | WhatsApp: 57.4% detection rate
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx
from collections import defaultdict
import pandas as pd

from .base_loader import BaseGraphLoader, LoaderError

logger = logging.getLogger(__name__)


class ConversationAnalysisLoader(BaseGraphLoader):
    """
    Load conversation analysis data with Phase 2 metadata.

    Converts ConvoKit corpus (31,249 messages, REALTALK + WhatsApp)
    to NetworkX MultiDiGraph with:
    - Person nodes (speakers with aggregated metrics)
    - Message nodes (with classification/topic/commitment metadata)
    - SENT edges (person → message)
    - REPLY_TO edges (message → message)
    - TALKS_TO edges (person → person, aggregated communication patterns)
    - TOPIC_CONTINUITY edges (message → message, same topic)
    - COMMITMENT edges (person → person, commitment flows)

    Node Types:
      - person: Speakers with aggregated communication metrics
      - message: Individual messages with Phase 2 semantic metadata

    Edge Types:
      - SENT: Who sent which message (person → message)
      - REPLY_TO: Message reply chains (message → message)
      - TALKS_TO: Aggregated communication (person → person)
      - TOPIC_CONTINUITY: Same-topic message flows (message → message)
      - COMMITMENT: Commitment flows between speakers (person → person)

    Phase 2 Metadata (Notebook 7 Updates):
      - Topic modeling uses separate BERTopic models per dataset:
        * REALTALK: 25 topics with English vectorizer
        * WhatsApp: 94 topics with Dutch stopwords vectorizer
      - Commitment detection uses separate language models:
        * REALTALK: English model (91.5% detection rate, 0.60 threshold)
        * WhatsApp: Dutch model (57.4% detection rate, 0.60 threshold)
        * All commitment types normalized to English labels
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize conversation loader.
        
        Config options:
          - corpus_path: Path to ConvoKit corpus directory
        """
        super().__init__(config)
        
        self.corpus_path = self.config.get(
            'corpus_path',
            '../conversation-analysis/data/processed/combined_bert'
        )
        
        # Load analysis results (communities and centrality)
        self.analysis_results_path = Path(self.corpus_path).parent.parent / 'analysis_results'
        self.communities = self._load_communities()
        self.centrality = self._load_centrality()
        
        self.logger.info(f"Conversation loader initialized: {self.corpus_path}")
        if self.communities:
            self.logger.info(f"   Loaded {len(self.communities)} community assignments")
        if self.centrality:
            self.logger.info(f"   Loaded centrality metrics for {len(self.centrality)} people")
    
    def _load_communities(self) -> Dict[str, int]:
        """Load community assignments from notebook 09 results"""
        try:
            csv_path = self.analysis_results_path / 'community_assignments.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                return dict(zip(df['person'], df['community']))
            else:
                self.logger.warning(f"Community assignments not found: {csv_path}")
                return {}
        except Exception as e:
            self.logger.warning(f"Could not load community assignments: {e}")
            return {}
    
    def _load_centrality(self) -> Dict[str, Dict[str, float]]:
        """Load centrality metrics from notebook 09 results"""
        try:
            csv_path = self.analysis_results_path / 'centrality_rankings.csv'
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                centrality_dict = {}
                for _, row in df.iterrows():
                    person = row['Person']  # Capital P
                    centrality_dict[person] = {
                        'pagerank_centrality': float(row.get('PageRank', 0.0)),
                        'betweenness_centrality': float(row.get('Betweenness', 0.0)),
                        'degree_centrality': float(row.get('Degree', 0.0)),
                    }
                return centrality_dict
            else:
                self.logger.warning(f"Centrality rankings not found: {csv_path}")
                return {}
        except Exception as e:
            self.logger.warning(f"Could not load centrality rankings: {e}")
            return {}
    
    def load_graph(self) -> nx.MultiDiGraph:
        """Load ConvoKit corpus and convert to NetworkX graph"""
        try:
            # Import ConvoKit here (not at module level to avoid dependency issues)
            from convokit import Corpus
            
            self.logger.info(f"Loading ConvoKit corpus from {self.corpus_path}...")
            corpus = Corpus(self.corpus_path)
            
            num_messages = len(list(corpus.iter_utterances()))
            self.logger.info(f"Loaded {num_messages:,} messages from corpus")
            
            # Build graph
            graph = nx.MultiDiGraph()
            
            # Add speaker nodes with aggregated metrics
            self.logger.info("Adding speaker nodes...")
            self._add_speaker_nodes(corpus, graph)
            
            # Add message nodes with Phase 2 metadata
            self.logger.info("Adding message nodes with Phase 2 metadata...")
            self._add_message_nodes(corpus, graph)
            
            # Add aggregated person-to-person relationships
            self.logger.info("Creating TALKS_TO edges (aggregated communication)...")
            self._add_talks_to_edges(corpus, graph)
            
            # Add topic continuity edges
            self.logger.info("Creating TOPIC_CONTINUITY edges...")
            self._add_topic_edges(corpus, graph)
            
            # Add commitment flows
            self.logger.info("Creating COMMITMENT edges...")
            self._add_commitment_edges(corpus, graph)
            
            self.logger.info(
                f"✅ Graph built: {graph.number_of_nodes():,} nodes, "
                f"{graph.number_of_edges():,} edges"
            )
            
            return graph
            
        except ImportError as e:
            self.logger.error("ConvoKit not installed. Please install: pip install convokit")
            raise LoaderError(f"ConvoKit dependency missing: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load conversation data: {e}")
            raise LoaderError(f"Conversation data loading failed: {e}")
    
    def _add_speaker_nodes(self, corpus, graph: nx.MultiDiGraph):
        """Add person nodes with aggregated communication metrics"""
        for speaker in corpus.iter_speakers():
            utts = list(speaker.iter_utterances())
            
            # Get community and centrality data
            community = self.communities.get(speaker.id)
            centrality_data = self.centrality.get(speaker.id, {})
            
            graph.add_node(
                speaker.id,
                node_type='person',
                name=speaker.id,
                total_messages=len(utts),
                question_ratio=self._calc_question_ratio(utts),
                answer_ratio=self._calc_answer_ratio(utts),
                commitment_count=self._calc_commitment_count(utts),
                avg_msg_length=self._calc_avg_length(utts),
                dataset=self._get_dataset(utts),
                unique_topics=self._count_unique_topics(utts),
                # Phase 3 metadata - Network Analysis (from notebook 09)
                community=community,
                pagerank_centrality=centrality_data.get('pagerank_centrality'),
                betweenness_centrality=centrality_data.get('betweenness_centrality'),
                degree_centrality=centrality_data.get('degree_centrality'),
            )
    
    def _add_message_nodes(self, corpus, graph: nx.MultiDiGraph):
        """Add message nodes with Phase 2 metadata"""
        for utt in corpus.iter_utterances():
            graph.add_node(
                utt.id,
                node_type='message',
                text=utt.text[:200],  # Truncate for graph storage
                full_text_length=len(utt.text),
                timestamp=str(utt.timestamp) if utt.timestamp else None,
                speaker=utt.speaker.id,
                conversation_id=utt.conversation_id,
                
                # Phase 2 metadata - Message Classification (Part 1.1, from notebook 06)
                message_type=utt.meta.get('message_type', 'neither'),
                is_question=utt.meta.get('question', False),
                is_answer=utt.meta.get('is_answer', False),
                
                # Phase 2 metadata - Topic Continuity (Part 1.2, from notebook 07)
                # Separate models: REALTALK (25 topics) + WhatsApp (94 topics) = 119 total
                topic_id=utt.meta.get('topic_id', -1),
                topic_label=utt.meta.get('topic_label', 'Unknown'),
                topic_shift=utt.meta.get('topic_shift', False),
                topic_segment_id=utt.meta.get('topic_segment_id', 0),
                
                # Phase 2 metadata - Commitment Detection (Part 1.3, from notebook 07)
                # Separate language models: EN (91.5% detection) + NL (57.4% detection)
                # Threshold: 0.60 confidence, types normalized to English
                has_commitment=utt.meta.get('has_commitment_bertopic', False),
                commitment_type=utt.meta.get('commitment_type_bertopic', None),
                commitment_confidence=utt.meta.get('commitment_confidence_bertopic', 0.0),
            )
            
            # SENT edge (person → message)
            graph.add_edge(
                utt.speaker.id,
                utt.id,
                relationship_type='SENT',
                timestamp=str(utt.timestamp) if utt.timestamp else None
            )
            
            # REPLY_TO edge (message → message)
            if utt.reply_to:
                graph.add_edge(
                    utt.id,
                    utt.reply_to,
                    relationship_type='REPLY_TO'
                )
    
    def _add_talks_to_edges(self, corpus, graph: nx.MultiDiGraph):
        """Create aggregated person-to-person communication edges"""
        interactions = defaultdict(lambda: {
            'frequency': 0,
            'questions': 0,
            'answers': 0,
            'commitments': 0,
            'topics': set(),
            'response_times': []
        })
        
        for utt in corpus.iter_utterances():
            if utt.reply_to:
                parent = corpus.get_utterance(utt.reply_to)
                if parent and parent.speaker.id != utt.speaker.id:
                    key = (parent.speaker.id, utt.speaker.id)
                    
                    interactions[key]['frequency'] += 1

                    if utt.meta.get('question', False):
                        interactions[key]['questions'] += 1
                    if utt.meta.get('is_answer', False):
                        interactions[key]['answers'] += 1
                    if utt.meta.get('has_commitment_bertopic', False):
                        interactions[key]['commitments'] += 1
                    
                    # Track shared topics
                    topic_id = utt.meta.get('topic_id', -1)
                    if topic_id >= 0:
                        interactions[key]['topics'].add(topic_id)
                    
                    # Calculate response time if timestamps available
                    if parent.timestamp and utt.timestamp:
                        time_diff = abs(utt.timestamp - parent.timestamp)
                        interactions[key]['response_times'].append(time_diff)
        
        # Add edges to graph with relationship classification
        for (speaker1, speaker2), data in interactions.items():
            avg_response = (
                sum(data['response_times']) / len(data['response_times'])
                if data['response_times'] else 0
            )
            
            # Check reciprocity (bidirectional communication)
            reverse_key = (speaker2, speaker1)
            reciprocity = reverse_key in interactions
            
            graph.add_edge(
                speaker1,
                speaker2,
                relationship_type='TALKS_TO',
                frequency=data['frequency'],
                questions_asked=data['questions'],
                answers_given=data['answers'],
                commitments_made=data['commitments'],
                shared_topics=len(data['topics']),
                avg_response_time=avg_response,
                reciprocity=reciprocity,
                relationship_class=self._classify_relationship(data, reciprocity)
            )
    
    def _add_topic_edges(self, corpus, graph: nx.MultiDiGraph):
        """Connect messages on same topic (topic continuity chains)"""
        for conv in corpus.iter_conversations():
            utts = sorted(
                list(conv.iter_utterances()), 
                key=lambda u: u.timestamp or 0
            )
            
            for i in range(len(utts) - 1):
                curr_topic = utts[i].meta.get('topic_id', -1)
                next_topic = utts[i+1].meta.get('topic_id', -1)
                
                # Same topic = topic continuity
                if curr_topic >= 0 and curr_topic == next_topic:
                    graph.add_edge(
                        utts[i].id,
                        utts[i+1].id,
                        relationship_type='TOPIC_CONTINUITY',
                        topic_id=curr_topic,
                        topic_label=utts[i].meta.get('topic_label', 'Unknown')
                    )
    
    def _add_commitment_edges(self, corpus, graph: nx.MultiDiGraph):
        """Track commitment flows between speakers"""
        for utt in corpus.iter_utterances():
            if utt.meta.get('has_commitment_bertopic', False) and utt.reply_to:
                parent = corpus.get_utterance(utt.reply_to)
                if parent and parent.speaker.id != utt.speaker.id:
                    graph.add_edge(
                        utt.speaker.id,
                        parent.speaker.id,
                        relationship_type='COMMITMENT',
                        commitment_type=utt.meta.get('commitment_type_bertopic'),
                        confidence=utt.meta.get('commitment_confidence_bertopic', 0.0),
                        message_id=utt.id
                    )
    
    def _classify_relationship(self, data: Dict, reciprocity: bool) -> str:
        """
        Classify relationship type based on interaction patterns.
        
        Relationship types:
          - casual: Low frequency (< 5 messages)
          - help_seeking: High question ratio (> 50% questions)
          - supportive: High answer ratio (> 50% answers)
          - planning: High commitment ratio (> 20% commitments)
          - collaborative: Reciprocal with balanced interactions
          - peer: High frequency, reciprocal, balanced
          - hierarchical: High frequency, not reciprocal
        """
        freq = data['frequency']
        
        if freq < 5:
            return 'casual'
        elif data['questions'] > freq * 0.5:
            return 'collaborative' if reciprocity else 'help_seeking'
        elif data['answers'] > freq * 0.5:
            return 'supportive'
        elif data['commitments'] > freq * 0.2:
            return 'planning'
        elif reciprocity:
            return 'peer'
        else:
            return 'hierarchical'
    
    # Helper methods for node attribute calculation
    
    def _calc_question_ratio(self, utts) -> float:
        """Calculate proportion of messages that are questions"""
        if not utts:
            return 0.0
        questions = sum(1 for u in utts if u.meta.get('question', False))
        return round(questions / len(utts), 3)
    
    def _calc_answer_ratio(self, utts) -> float:
        """Calculate proportion of messages that are answers"""
        if not utts:
            return 0.0
        answers = sum(1 for u in utts if u.meta.get('is_answer', False))
        return round(answers / len(utts), 3)
    
    def _calc_commitment_count(self, utts) -> int:
        """Count total commitments made by speaker"""
        return sum(1 for u in utts if u.meta.get('has_commitment_bertopic', False))
    
    def _calc_avg_length(self, utts) -> float:
        """Calculate average message length"""
        if not utts:
            return 0.0
        return round(sum(len(u.text) for u in utts) / len(utts), 1)
    
    def _get_dataset(self, utts) -> str:
        """Get dataset name (REALTALK or WhatsApp)"""
        for utt in utts:
            return utt.meta.get('dataset', 'unknown')
        return 'unknown'
    
    def _count_unique_topics(self, utts) -> int:
        """Count unique topics discussed by speaker"""
        topics = set()
        for utt in utts:
            topic_id = utt.meta.get('topic_id', -1)
            if topic_id >= 0:
                topics.add(topic_id)
        return len(topics)
    
    # Required BaseGraphLoader methods
    
    def get_dataset_description(self) -> str:
        """Get human-readable description of this dataset"""
        return (
            "Multilingual conversation analysis dataset with 31,249 messages from "
            "REALTALK (English, 8,944 messages) and WhatsApp (Dutch, 22,305 messages) corpora. "
            "Enriched with Phase 2 semantic analysis using separate language-specific models: "
            "message classification (95%+ accuracy, 16.0% questions, 23.1% answers), "
            "topic modeling (119 topics: 25 REALTALK + 94 WhatsApp, 80.6% topic shifts), "
            "and commitment detection (20,983 commitments, 67.1% detection rate: "
            "REALTALK 91.5%, WhatsApp 57.4%). "
            "Data includes speaker relationships, conversation dynamics, and topic continuity "
            "patterns for relationship extraction and network analysis."
        )
    
    def get_sample_queries(self) -> List[str]:
        """Get domain-appropriate sample questions"""
        return [
            # Relationship analysis
            "Who are the most active communicators in the network?",
            "Find communities of people who communicate frequently",
            "Which speakers have reciprocal relationships?",
            "Show me help-seeking vs supportive relationships",
            
            # Topic-based queries
            "What topics do speakers discuss most?",
            "Which speakers share common topics?",
            "Find conversations with frequent topic shifts",
            "Show me topic continuity patterns",
            
            # Commitment analysis
            "Who makes the most commitments?",
            "Find commitment flows between speakers",
            "What types of commitments are most common?",
            "Show me planning relationships based on commitments",
            
            # Question/Answer dynamics
            "Which speakers ask the most questions?",
            "Who provides the most answers?",
            "Find question-answer pairs in conversations",
            "Identify collaborative relationships",
            
            # Network analysis
            "Calculate PageRank centrality for speakers",
            "Detect communities using Louvain algorithm",
            "Find the most influential speakers",
            "What's the network density and clustering coefficient?",
            
            # Relationship classification
            "Classify relationships by interaction patterns",
            "Find hierarchical vs peer relationships",
            "Show me casual vs intense communication patterns",
            "Which speakers have the most diverse connections?"
        ]
    
    def get_dataset_name(self) -> str:
        """Get dataset name"""
        return "Conversation Analysis (REALTALK + WhatsApp)"
    
    def get_data_sources(self) -> List[str]:
        """Get list of data source paths"""
        return [str(Path(self.corpus_path).resolve())]
