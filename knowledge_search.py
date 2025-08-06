import json
import re
from typing import Dict, List, Any, Tuple
from difflib import SequenceMatcher
import streamlit as st

class TourismKnowledgeBase:
    def __init__(self, kb_file_path: str = "andhra_tourism_kb.json"):
        """Initialize the knowledge base with advanced search capabilities"""
        self.kb_data = self.load_knowledge_base(kb_file_path)
        self.search_index = self.build_search_index()
        
    def load_knowledge_base(self, file_path: str) -> Dict:
        """Load knowledge base from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            st.error(f"Knowledge base file {file_path} not found!")
            return {}
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in {file_path}")
            return {}
    
    def build_search_index(self) -> Dict[str, List[str]]:
        """Build a comprehensive search index for faster lookups"""
        index = {}
        
        def add_to_index(text: str, path: str):
            """Add text and its variations to search index"""
            if not text:
                return
                
            # Convert to lowercase and split into words
            words = re.findall(r'\b\w+\b', text.lower())
            
            for word in words:
                if len(word) > 2:  # Ignore very short words
                    if word not in index:
                        index[word] = []
                    if path not in index[word]:
                        index[word].append(path)
        
        def index_recursive(data: Any, path: str = ""):
            """Recursively index all text content"""
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    add_to_index(key, current_path)
                    index_recursive(value, current_path)
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]"
                    index_recursive(item, current_path)
            elif isinstance(data, str):
                add_to_index(data, path)
        
        index_recursive(self.kb_data)
        return index
    
    def smart_search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Advanced search with multiple techniques and enhanced intelligence"""
        query = query.lower().strip()
        results = []
        
        # Enhanced query preprocessing
        processed_query = self.preprocess_query(query)
        
        # 1. Intent detection
        intent = self.detect_intent(processed_query)
        
        # 2. Exact phrase matching
        exact_matches = self.exact_phrase_search(processed_query)
        results.extend(exact_matches)
        
        # 3. Keyword-based search with synonyms
        keyword_matches = self.enhanced_keyword_search(processed_query)
        results.extend(keyword_matches)
        
        # 4. Fuzzy matching for typos
        fuzzy_matches = self.fuzzy_search(processed_query)
        results.extend(fuzzy_matches)
        
        # 5. Semantic search (category-based)
        semantic_matches = self.semantic_search(processed_query)
        results.extend(semantic_matches)
        
        # 6. Context-aware search based on intent
        context_matches = self.context_aware_search(processed_query, intent)
        results.extend(context_matches)
        
        # Remove duplicates and rank results with intelligence
        unique_results = self.intelligent_ranking(results, processed_query, intent)
        
        return unique_results[:max_results]
    
    def exact_phrase_search(self, query: str) -> List[Dict[str, Any]]:
        """Search for exact phrases in the knowledge base"""
        results = []
        
        def search_in_data(data: Any, path: str = "", parent_context: str = ""):
            if isinstance(data, dict):
                for key, value in data.items():
                    current_path = f"{path}.{key}" if path else key
                    current_context = f"{parent_context} {key}" if parent_context else key
                    
                    if query in key.lower():
                        results.append({
                            'type': 'exact_match',
                            'path': current_path,
                            'content': value,
                            'context': current_context,
                            'match_location': 'key',
                            'relevance_score': 1.0
                        })
                    
                    search_in_data(value, current_path, current_context)
            
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    current_path = f"{path}[{i}]"
                    search_in_data(item, current_path, parent_context)
            
            elif isinstance(data, str) and query in data.lower():
                results.append({
                    'type': 'exact_match',
                    'path': path,
                    'content': data,
                    'context': parent_context,
                    'match_location': 'content',
                    'relevance_score': 0.9
                })
        
        search_in_data(self.kb_data)
        return results
    
    def keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Search based on individual keywords"""
        query_words = re.findall(r'\b\w+\b', query.lower())
        results = []
        
        for word in query_words:
            if word in self.search_index:
                for path in self.search_index[word]:
                    content = self.get_content_by_path(path)
                    if content:
                        results.append({
                            'type': 'keyword_match',
                            'path': path,
                            'content': content,
                            'context': path.split('.')[-1],
                            'match_word': word,
                            'relevance_score': 0.7
                        })
        
        return results
    
    def fuzzy_search(self, query: str) -> List[Dict[str, Any]]:
        """Search with fuzzy matching for typos and similar words"""
        results = []
        query_words = re.findall(r'\b\w+\b', query.lower())
        
        for query_word in query_words:
            for indexed_word in self.search_index.keys():
                similarity = SequenceMatcher(None, query_word, indexed_word).ratio()
                
                if similarity > 0.8:  # 80% similarity threshold
                    for path in self.search_index[indexed_word]:
                        content = self.get_content_by_path(path)
                        if content:
                            results.append({
                                'type': 'fuzzy_match',
                                'path': path,
                                'content': content,
                                'context': path.split('.')[-1],
                                'match_word': indexed_word,
                                'query_word': query_word,
                                'similarity': similarity,
                                'relevance_score': similarity * 0.6
                            })
        
        return results
    
    def semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Search based on semantic categories and synonyms"""
        results = []
        
        # Define semantic categories with synonyms
        semantic_categories = {
            'destinations': ['place', 'location', 'city', 'town', 'visit', 'go', 'destination', 'tourist', 'sightseeing'],
            'temples': ['temple', 'shrine', 'worship', 'religious', 'pilgrimage', 'devotion', 'darshan', 'god', 'deity', 'sacred'],
            'food': ['food', 'eat', 'cuisine', 'dish', 'restaurant', 'meal', 'cooking', 'recipe', 'taste', 'flavor'],
            'transportation': ['transport', 'travel', 'reach', 'train', 'flight', 'bus', 'road', 'airport', 'station'],
            'accommodation': ['hotel', 'stay', 'accommodation', 'lodge', 'resort', 'room', 'booking'],
            'festivals': ['festival', 'celebration', 'event', 'cultural', 'tradition', 'ceremony'],
            'adventure': ['adventure', 'sports', 'trekking', 'water', 'wildlife', 'safari', 'activities']
        }
        
        query_lower = query.lower()
        
        for category, keywords in semantic_categories.items():
            category_score = sum(1 for keyword in keywords if keyword in query_lower)
            
            if category_score > 0:
                if category in self.kb_data:
                    results.append({
                        'type': 'semantic_match',
                        'path': category,
                        'content': self.kb_data[category],
                        'context': category,
                        'category': category,
                        'relevance_score': min(category_score * 0.5, 0.8)
                    })
        
        return results
    
    def get_content_by_path(self, path: str) -> Any:
        """Retrieve content from knowledge base using dot notation path"""
        try:
            parts = path.split('.')
            current = self.kb_data
            
            for part in parts:
                if '[' in part and ']' in part:
                    # Handle array indices
                    key, index_str = part.split('[')
                    index = int(index_str.rstrip(']'))
                    current = current[key][index]
                else:
                    current = current[part]
            
            return current
        except (KeyError, IndexError, ValueError):
            return None
    
    def preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing"""
        # Handle common variations and synonyms
        synonyms = {
            'vizag': 'visakhapatnam',
            'hyd': 'hyderabad',
            'ap': 'andhra pradesh',
            'temple': 'temple worship religious',
            'food': 'cuisine dish meal restaurant',
            'travel': 'transport reach journey',
            'stay': 'accommodation hotel lodge',
            'cost': 'price budget money expense',
            'weather': 'climate season temperature'
        }
        
        processed = query
        for synonym, expansion in synonyms.items():
            if synonym in processed:
                processed = processed.replace(synonym, expansion)
        
        return processed
    
    def detect_intent(self, query: str) -> str:
        """Detect user intent from query"""
        intents = {
            'destination_info': ['about', 'tell me', 'information', 'describe', 'what is'],
            'travel_planning': ['how to reach', 'travel to', 'journey', 'route', 'transport'],
            'accommodation': ['stay', 'hotel', 'lodge', 'accommodation', 'where to stay'],
            'food_query': ['food', 'eat', 'restaurant', 'cuisine', 'dish', 'meal'],
            'cost_inquiry': ['cost', 'price', 'budget', 'expense', 'how much'],
            'timing': ['when', 'time', 'season', 'weather', 'best time'],
            'comparison': ['vs', 'versus', 'compare', 'difference', 'better'],
            'recommendation': ['best', 'top', 'recommend', 'suggest', 'popular']
        }
        
        for intent, keywords in intents.items():
            if any(keyword in query for keyword in keywords):
                return intent
        
        return 'general_query'
    
    def enhanced_keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Enhanced keyword search with synonyms and context"""
        results = []
        query_words = re.findall(r'\b\w+\b', query.lower())
        
        # Add related terms based on context
        expanded_words = set(query_words)
        for word in query_words:
            if word in ['temple', 'worship']:
                expanded_words.update(['religious', 'pilgrimage', 'deity', 'darshan'])
            elif word in ['food', 'eat']:
                expanded_words.update(['cuisine', 'restaurant', 'dish', 'meal'])
            elif word in ['travel', 'reach']:
                expanded_words.update(['transport', 'journey', 'route'])
        
        for word in expanded_words:
            if word in self.search_index:
                for path in self.search_index[word]:
                    content = self.get_content_by_path(path)
                    if content:
                        results.append({
                            'type': 'enhanced_keyword_match',
                            'path': path,
                            'content': content,
                            'context': path.split('.')[-1],
                            'match_word': word,
                            'relevance_score': 0.8 if word in query_words else 0.6
                        })
        
        return results
    
    def context_aware_search(self, query: str, intent: str) -> List[Dict[str, Any]]:
        """Search based on detected intent and context"""
        results = []
        
        if intent == 'travel_planning':
            # Focus on transportation and routes
            transport_data = self.kb_data.get('transportation', {})
            local_transport = self.kb_data.get('local_transportation', {})
            
            results.append({
                'type': 'context_aware',
                'path': 'transportation',
                'content': transport_data,
                'context': 'travel_planning',
                'relevance_score': 0.9
            })
        
        elif intent == 'accommodation':
            # Focus on accommodation details
            accommodation_data = self.kb_data.get('accommodation_details', {})
            results.append({
                'type': 'context_aware',
                'path': 'accommodation_details',
                'content': accommodation_data,
                'context': 'accommodation',
                'relevance_score': 0.9
            })
        
        elif intent == 'cost_inquiry':
            # Focus on budget and cost information
            for dest_key, dest_info in self.kb_data.get('destinations', {}).items():
                if 'budget' in dest_info:
                    results.append({
                        'type': 'context_aware',
                        'path': f'destinations.{dest_key}',
                        'content': dest_info,
                        'context': 'cost_inquiry',
                        'relevance_score': 0.8
                    })
        
        return results
    
    def intelligent_ranking(self, results: List[Dict[str, Any]], query: str, intent: str) -> List[Dict[str, Any]]:
        """Intelligent ranking based on query and intent"""
        # Remove duplicates based on path
        seen_paths = set()
        unique_results = []
        
        for result in results:
            if result['path'] not in seen_paths:
                seen_paths.add(result['path'])
                
                # Boost relevance based on intent matching
                if result.get('context') == intent:
                    result['relevance_score'] += 0.2
                
                # Boost exact matches
                if result.get('type') == 'exact_match':
                    result['relevance_score'] += 0.3
                
                # Boost recent or popular destinations
                if 'tirupati' in result.get('path', '').lower():
                    result['relevance_score'] += 0.1
                
                unique_results.append(result)
        
        # Sort by enhanced relevance score
        unique_results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return unique_results
    
    def deduplicate_and_rank(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Remove duplicates and rank results by relevance (legacy method)"""
        return self.intelligent_ranking(results, query, 'general_query')
    
    def format_search_result(self, result: Dict[str, Any]) -> str:
        """Format search result for display"""
        content = result['content']
        result_type = result['type']
        context = result.get('context', '')
        
        if isinstance(content, dict):
            return self.format_dict_content(content, context)
        elif isinstance(content, list):
            return self.format_list_content(content, context)
        else:
            return str(content)
    
    def format_dict_content(self, content: Dict, context: str) -> str:
        """Format dictionary content for display"""
        if 'name' in content and 'description' in content:
            # Destination or attraction format
            result = f"**{content['name']}**\n\n"
            result += f"{content['description']}\n\n"
            
            if 'attractions' in content:
                result += "**Key Attractions:**\n"
                attractions = content['attractions'][:4]  # Show top 4
                for attraction in attractions:
                    if isinstance(attraction, dict):
                        result += f"• **{attraction.get('name', 'Attraction')}** - {attraction.get('description', '')}\n"
                    else:
                        result += f"• {attraction}\n"
                result += "\n"
            
            if 'best_time' in content:
                result += f"**Best Time to Visit:** {content['best_time']}\n\n"
            
            if 'how_to_reach' in content:
                result += "**How to Reach:**\n"
                reach_info = content['how_to_reach']
                if isinstance(reach_info, dict):
                    for mode, info in reach_info.items():
                        result += f"• **{mode.replace('_', ' ').title()}:** {info}\n"
                result += "\n"
            
            if 'duration' in content:
                result += f"**Recommended Duration:** {content['duration']}\n\n"
            
            return result
        
        elif 'deity' in content and 'significance' in content:
            # Temple format
            result = f"**{content.get('name', 'Temple')}**\n\n"
            result += f"**Deity:** {content['deity']}\n"
            result += f"**Significance:** {content['significance']}\n\n"
            
            if 'timings' in content:
                result += f"**Timings:** {content['timings']}\n"
            
            if 'entry_fee' in content:
                result += f"**Entry Fee:** {content['entry_fee']}\n"
            
            if 'location' in content:
                result += f"**Location:** {content['location']}\n"
            
            return result
        
        else:
            # Generic dictionary format
            result = f"**{context.replace('_', ' ').title()}**\n\n"
            for key, value in list(content.items())[:5]:  # Show top 5 items
                if isinstance(value, (str, int, float)):
                    result += f"**{key.replace('_', ' ').title()}:** {value}\n"
            return result
    
    def format_list_content(self, content: List, context: str) -> str:
        """Format list content for display"""
        result = f"**{context.replace('_', ' ').title()}**\n\n"
        
        for i, item in enumerate(content[:5]):  # Show top 5 items
            if isinstance(item, dict):
                name = item.get('name', f'Item {i+1}')
                desc = item.get('description', '')
                if desc:
                    result += f"**{name}** - {desc}\n"
                else:
                    result += f"**{name}**\n"
            else:
                result += f"• {item}\n"
        
        if len(content) > 5:
            result += f"\n*({len(content) - 5} more items available)*"
        
        return result
    
    def get_destination_info(self, destination_name: str) -> str:
        """Get comprehensive information about a specific destination"""
        destination_name = destination_name.lower()
        
        # Search in destinations
        destinations = self.kb_data.get('destinations', {})
        
        for key, dest_info in destinations.items():
            if (destination_name in key.lower() or 
                destination_name in dest_info.get('name', '').lower() or
                any(destination_name in alias.lower() for alias in dest_info.get('aliases', []))):
                
                return self.format_dict_content(dest_info, dest_info.get('name', key))
        
        return f"Sorry, I couldn't find detailed information about {destination_name}. Try asking about popular destinations like Tirupati, Visakhapatnam, or Araku Valley."
    
    def get_temple_info(self, temple_query: str) -> str:
        """Get information about temples"""
        temple_query = temple_query.lower()
        
        temples = self.kb_data.get('temples', {}).get('major_temples', [])
        
        for temple in temples:
            if (temple_query in temple.get('name', '').lower() or
                temple_query in temple.get('location', '').lower() or
                temple_query in temple.get('deity', '').lower()):
                
                return self.format_dict_content(temple, temple.get('name', 'Temple'))
        
        # Return general temple information
        result = "**Major Temples of Andhra Pradesh**\n\n"
        for temple in temples[:4]:
            result += f"**{temple.get('name')}** ({temple.get('location')})\n"
            result += f"*Deity: {temple.get('deity')}*\n\n"
        
        result += "*Ask about specific temples like 'Tirupati temple' or 'Simhachalam temple' for detailed information.*"
        return result
    
    def get_food_info(self, food_query: str) -> str:
        """Get information about Andhra Pradesh cuisine"""
        food_query = food_query.lower()
        
        food_data = self.kb_data.get('food_culture', {})
        
        if 'biryani' in food_query:
            # Specific biryani information
            regional_cuisines = food_data.get('regional_cuisines', {})
            hyderabad_cuisine = regional_cuisines.get('hyderabad_nizami', {})
            
            for dish in hyderabad_cuisine.get('signature_dishes', []):
                if 'biryani' in dish.get('name', '').lower():
                    result = f"**{dish['name']}**\n\n"
                    result += f"{dish['description']}\n\n"
                    result += f"**History:** {dish.get('history', 'Ancient recipe')}\n"
                    result += f"**Cooking Method:** {dish.get('cooking_method', 'Traditional dum cooking')}\n\n"
                    
                    if 'served_with' in dish:
                        result += "**Served with:** " + ", ".join(dish['served_with']) + "\n\n"
                    
                    # Add restaurant information
                    restaurants = food_data.get('famous_restaurants', [])
                    biryani_restaurants = [r for r in restaurants if 'biryani' in r.get('specialty', '').lower()]
                    
                    if biryani_restaurants:
                        result += "**Best Places to Try:**\n"
                        for restaurant in biryani_restaurants[:3]:
                            result += f"• **{restaurant['name']}** ({restaurant.get('location', '')})\n"
                    
                    return result
        
        # General food information
        result = "**Andhra Pradesh Cuisine**\n\n"
        result += "Known for spicy and flavorful dishes with distinct regional variations.\n\n"
        
        # Regional cuisines
        regional_cuisines = food_data.get('regional_cuisines', {})
        for region, cuisine_info in regional_cuisines.items():
            result += f"**{region.replace('_', ' ').title()}:**\n"
            characteristics = cuisine_info.get('characteristics', [])
            if characteristics:
                result += f"*{', '.join(characteristics)}*\n"
            
            signature_dishes = cuisine_info.get('signature_dishes', [])[:2]
            for dish in signature_dishes:
                result += f"• **{dish.get('name')}** - {dish.get('description')}\n"
            result += "\n"
        
        return result

class HybridAIEngine:
    """
    Hybrid AI Engine integrated with Knowledge Base
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.ai_available = False
        self.ai_pipeline = None
        self.sentiment_pipeline = None
        self.initialize_ai()
    
    def initialize_ai(self):
        """Try to initialize AI models with graceful fallback"""
        try:
            from transformers import pipeline
            
            # Try to load sentiment analysis (more reliable)
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            
            # Try QA model (may fail due to auth issues)
            try:
                self.ai_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-cased-distilled-squad"
                )
                self.ai_available = True
            except:
                self.ai_available = False
                
        except:
            self.ai_available = False
    
    def get_intelligent_response(self, query: str) -> dict:
        """Get intelligent response using AI + Knowledge Base hybrid"""
        response_data = {
            "answer": "",
            "source": "Enhanced Knowledge Base",
            "confidence": 0.8,
            "suggestions": [],
            "sentiment": "neutral"
        }
        
        # Analyze sentiment
        sentiment = self.analyze_sentiment(query)
        response_data["sentiment"] = sentiment["sentiment"]
        
        # Try AI first if available
        if self.ai_available and self.ai_pipeline:
            ai_response = self.try_ai_response(query)
            if ai_response["success"] and ai_response["confidence"] > 0.3:
                response_data["answer"] = ai_response["answer"]
                response_data["source"] = "AI + Knowledge Base"
                response_data["confidence"] = ai_response["confidence"]
                
                # Enhance with knowledge base
                kb_context = self.get_knowledge_response(query)
                if kb_context:
                    response_data["answer"] += f"\n\n**Additional Details:**\n{kb_context}"
                
                response_data["suggestions"] = self.generate_suggestions(query)
                return response_data
        
        # Fallback to knowledge base
        kb_response = self.get_knowledge_response(query)
        if kb_response:
            response_data["answer"] = kb_response
        else:
            response_data["answer"] = self.get_simple_response(query)
            response_data["source"] = "Simple AI"
            response_data["confidence"] = 0.6
        
        response_data["suggestions"] = self.generate_suggestions(query)
        return response_data
    
    def try_ai_response(self, query: str) -> dict:
        """Try AI response with error handling"""
        try:
            # Get context from knowledge base
            search_results = self.kb.smart_search(query, max_results=1)
            if not search_results:
                return {"success": False, "answer": "", "confidence": 0.0}
            
            # Extract context
            result = search_results[0]
            if isinstance(result.get('content'), dict):
                content = result['content']
                context = content.get('description', str(content)[:500])
            else:
                context = str(result.get('content', ''))[:500]
            
            # Use AI
            ai_result = self.ai_pipeline(question=query, context=context)
            return {
                "success": True,
                "answer": ai_result["answer"],
                "confidence": ai_result["score"]
            }
        except:
            return {"success": False, "answer": "", "confidence": 0.0}
    
    def get_knowledge_response(self, query: str) -> str:
        """Get response from knowledge base"""
        try:
            search_results = self.kb.smart_search(query, max_results=1)
            if search_results:
                return self.kb.format_search_result(search_results[0])
            return ""
        except:
            return ""
    
    def get_simple_response(self, query: str) -> str:
        """Simple fallback responses"""
        query_lower = query.lower()
        
        responses = {
            'tirupati': "Tirupati is famous for the Tirumala Venkateswara Temple, one of the richest temples in the world.",
            'visakhapatnam': "Visakhapatnam is a major port city known for beautiful beaches and the gateway to Araku Valley.",
            'vizag': "Vizag offers beautiful beaches, submarine museum, and scenic Araku Valley nearby.",
            'hyderabad': "Hyderabad is famous for Charminar, Golconda Fort, and world-renowned Hyderabadi Biryani.",
            'araku': "Araku Valley is a hill station famous for coffee plantations and scenic train journey.",
            'temple': "Andhra Pradesh has many famous temples. Tirupati is the most visited pilgrimage site.",
            'food': "Andhra Pradesh cuisine is known for spiciness. Famous for Hyderabadi Biryani and traditional dishes.",
            'biryani': "Hyderabadi Biryani is world-famous. Best places: Paradise Restaurant, Hotel Shadab, Bawarchi."
        }
        
        for keyword, response in responses.items():
            if keyword in query_lower:
                return response
        
        return "I can help you with information about Andhra Pradesh destinations, temples, food, and travel. Please ask me something specific!"
    
    def analyze_sentiment(self, text: str) -> dict:
        """Analyze sentiment with fallback"""
        try:
            if self.sentiment_pipeline:
                result = self.sentiment_pipeline(text)[0]
                return {
                    "sentiment": result["label"].lower(),
                    "confidence": result["score"]
                }
        except:
            pass
        
        # Simple fallback sentiment analysis
        positive_words = ['love', 'great', 'amazing', 'wonderful', 'excellent', 'good', 'best']
        negative_words = ['hate', 'terrible', 'awful', 'bad', 'worst', 'horrible']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return {"sentiment": "positive", "confidence": 0.7}
        elif neg_count > pos_count:
            return {"sentiment": "negative", "confidence": 0.7}
        else:
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def generate_suggestions(self, query: str) -> list:
        """Generate smart suggestions"""
        query_lower = query.lower()
        
        if 'tirupati' in query_lower:
            return [
                "What are the darshan booking procedures?",
                "Best accommodation near Tirumala?",
                "Temple festival calendar?"
            ]
        elif 'visakhapatnam' in query_lower or 'vizag' in query_lower:
            return [
                "Best beaches for water sports?",
                "Araku Valley train journey details?",
                "Submarine museum visiting hours?"
            ]
        elif 'food' in query_lower or 'biryani' in query_lower:
            return [
                "Vegetarian restaurant recommendations?",
                "Traditional breakfast options?",
                "Food safety tips for tourists?"
            ]
        elif 'temple' in query_lower:
            return [
                "Temple dress code guidelines?",
                "Photography rules in temples?",
                "Best time to avoid crowds?"
            ]
        else:
            return [
                "Tell me about top destinations",
                "What's the best time to visit AP?",
                "How to plan a 5-day itinerary?"
            ]

# Initialize the knowledge base
@st.cache_resource
def load_knowledge_base():
    """Load and cache the knowledge base"""
    return TourismKnowledgeBase()