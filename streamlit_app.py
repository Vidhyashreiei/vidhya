import streamlit as st
import json
import re
import difflib
from typing import Dict, List, Any, Tuple
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Saanchari - Andhra Pradesh Tourism Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import the advanced knowledge base with integrated hybrid AI
from knowledge_search import TourismKnowledgeBase, HybridAIEngine
class ChatEngine:
    def __init__(self):
        self.kb = TourismKnowledgeBase()
        self.ai_engine = HybridAIEngine(self.kb)
        self.conversation_history = []
        self.user_preferences = {}
        self.session_context = {}
        self.question_patterns = self.load_question_patterns()
        # Removed response cache to ensure fresh responses
        
    def get_response(self, user_input: str) -> str:
        user_input_lower = user_input.lower().strip()
        
        # Add to conversation history
        self.conversation_history.append(user_input)
        
        # Extract user preferences
        self.extract_preferences(user_input_lower)
        
        # Handle greetings with personalization
        if self.is_greeting(user_input_lower):
            return self.get_personalized_greeting()
        
        # Handle goodbyes
        elif self.is_goodbye(user_input_lower):
            return self.get_goodbye_response()
        
        # Handle follow-up questions
        elif self.is_followup_question(user_input_lower):
            return self.handle_followup(user_input_lower)
        
        # Use Hybrid AI Engine for intelligent responses
        ai_response = self.ai_engine.get_intelligent_response(user_input)
        
        # Format the response with metadata
        response = self.format_ai_response(ai_response, user_input)
        
        # Add personalized suggestions
        response = self.add_personalized_suggestions(response)
        
        return response
    
    def is_greeting(self, text: str) -> bool:
        greetings = ['hello', 'hi', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening']
        return any(greeting in text for greeting in greetings)
    
    def is_goodbye(self, text: str) -> bool:
        goodbyes = ['bye', 'goodbye', 'see you', 'thank you', 'thanks', 'dhanyawad']
        return any(goodbye in text for goodbye in goodbyes)
    
    def get_greeting_response(self) -> str:
        return ("**Welcome to Saanchari!**\n\nYour comprehensive Andhra Pradesh tourism guide. "
                "I can help you with destinations, temples, food, travel routes, accommodations, and custom itineraries.\n\n"
                "What would you like to explore in Andhra Pradesh?")
    
    def get_goodbye_response(self) -> str:
        return ("**Thank you for using Saanchari!**\n\n"
                "Have a wonderful time exploring Andhra Pradesh. Don't forget to try the delicious biryani and visit the magnificent temples.\n\n"
                "Safe travels and come back anytime for more tourism guidance!")
    
    def process_query(self, query: str) -> str:
        query_lower = query.lower()
        
        # Check for specific destination queries first
        destinations = ['tirupati', 'visakhapatnam', 'vizag', 'araku', 'vijayawada', 'hyderabad', 'srisailam']
        for dest in destinations:
            if dest in query_lower:
                return self.kb.get_destination_info(dest)
        
        # Check for temple queries
        temple_keywords = ['temple', 'worship', 'religious', 'pilgrimage', 'darshan', 'tirumala', 'simhachalam', 'kanaka durga']
        if any(keyword in query_lower for keyword in temple_keywords):
            return self.kb.get_temple_info(query)
        
        # Check for food queries
        food_keywords = ['food', 'cuisine', 'biryani', 'eat', 'restaurant', 'dish', 'meal']
        if any(keyword in query_lower for keyword in food_keywords):
            return self.kb.get_food_info(query)
        
        # Check for transportation queries
        transport_keywords = ['reach', 'transport', 'travel', 'train', 'flight', 'bus', 'airport', 'station']
        if any(keyword in query_lower for keyword in transport_keywords):
            return self.get_transport_info()
        
        # Check for timing/weather queries
        timing_keywords = ['time', 'weather', 'season', 'when', 'best time', 'climate']
        if any(keyword in query_lower for keyword in timing_keywords):
            return self.get_timing_info()
        
        # Check for destination list queries
        if any(word in query_lower for word in ['top', 'best', 'famous', 'popular', 'destinations', '10']):
            return self.get_top_destinations()
        
        # Use smart search as fallback
        search_results = self.kb.smart_search(query, max_results=1)
        if search_results:
            return self.kb.format_search_result(search_results[0])
        
        return ""
    
    def get_destinations_info(self, query: str) -> str:
        # Check for specific destination first
        specific_info = self.kb.get_destination_info(query)
        if "Sorry, I couldn't find" not in specific_info:
            return specific_info
        
        # General destinations query
        if any(word in query for word in ['top', 'best', 'famous', 'popular', 'all', '10']):
            return self.get_top_destinations()
        
        return self.get_general_destinations()
    
    def format_destination_info(self, dest_info: Dict) -> str:
        response = f"🏛️ **{dest_info['name']}**\n\n"
        response += f"📝 **About:** {dest_info['description']}\n\n"
        
        if 'attractions' in dest_info:
            response += "🎯 **Main Attractions:**\n"
            for attraction in dest_info['attractions']:
                response += f"• {attraction}\n"
            response += "\n"
        
        if 'best_time' in dest_info:
            response += f"🌤️ **Best Time to Visit:** {dest_info['best_time']}\n\n"
        
        if 'how_to_reach' in dest_info:
            response += "🚗 **How to Reach:**\n"
            for mode, info in dest_info['how_to_reach'].items():
                response += f"• **{mode.replace('_', ' ').title()}:** {info}\n"
            response += "\n"
        
        if 'duration' in dest_info:
            response += f"⏱️ **Recommended Duration:** {dest_info['duration']}\n\n"
        
        return response
    
    def get_top_destinations(self) -> str:
        return """**Top Destinations in Andhra Pradesh**

**1. Tirupati** - World famous Venkateswara Temple, spiritual capital
**2. Visakhapatnam (Vizag)** - Beautiful beaches, submarine museum, port city
**3. Araku Valley** - Hill station with coffee plantations and tribal culture
**4. Vijayawada** - Business capital, Kanaka Durga Temple, Krishna River
**5. Hyderabad** - Historic city, Charminar, Golconda Fort, IT hub
**6. Srisailam** - Jyotirlinga temple, wildlife sanctuary
**7. Amaravati** - Ancient Buddhist site, planned capital
**8. Lepakshi** - Historic temple with hanging pillar
**9. Gandikota** - Grand Canyon of India
**10. Horsley Hills** - Hill station, adventure activities

*Each destination offers unique experiences from spiritual journeys to adventure tourism.*"""
    
    def get_food_info(self, query: str) -> str:
        return self.kb.get_food_info(query)
    
    def get_temple_info(self, query: str) -> str:
        return self.kb.get_temple_info(query)
    
    def get_transport_info(self) -> str:
        return """**Transportation in Andhra Pradesh**

**By Air:**
• **Rajiv Gandhi International Airport (Hyderabad)** - Major international hub
• **Visakhapatnam Airport** - Coastal region gateway
• **Tirupati Airport** - Pilgrimage destination
• **Vijayawada Airport** - Business center

**By Train:**
• **Major stations:** Hyderabad, Vijayawada, Visakhapatnam, Tirupati
• **Special trains:** Vande Bharat Express, Rajdhani Express
• Well connected to all major Indian cities

**By Road:**
• **NH44 (Delhi-Chennai)** - Major north-south highway
• **NH16 (Kolkata-Chennai)** - Coastal highway
• **APSRTC** operates extensive bus network
• Car rental available in all major cities"""
    
    def get_timing_info(self) -> str:
        return """**Best Time to Visit Andhra Pradesh**

**Winter (November to February)**
• Pleasant weather (20-30°C)
• Ideal for sightseeing and temple visits
• Peak tourist season - book in advance

**Summer (March to May)**
• Hot weather (30-45°C)
• Good for hill stations like Araku Valley
• Off-season discounts available

**Monsoon (June to October)**
• Heavy rainfall, lush greenery
• Perfect for waterfalls and nature
• Some travel disruptions possible

**Recommended:** *October to March for most destinations*"""
    
    def get_general_destinations(self) -> str:
        return """**Andhra Pradesh Tourism Overview**

**Historical & Cultural:**
• **Hyderabad** - Charminar, Golconda Fort
• **Amaravati** - Buddhist heritage
• **Warangal** - Thousand Pillar Temple

**Spiritual Destinations:**
• **Tirupati** - Venkateswara Temple
• **Srisailam** - Jyotirlinga Temple
• **Vijayawada** - Kanaka Durga Temple

**Beach Destinations:**
• **Visakhapatnam** - RK Beach, Submarine Museum
• **Machilipatnam** - Historic port city

**Hill Stations:**
• **Araku Valley** - Coffee plantations
• **Horsley Hills** - Adventure activities

*Tell me which type of destination interests you most for detailed information!*"""
    
    def extract_preferences(self, user_input: str):
        """Extract user preferences from conversation"""
        if 'budget' in user_input or 'cheap' in user_input or 'affordable' in user_input:
            self.user_preferences['budget'] = 'budget'
        elif 'luxury' in user_input or 'expensive' in user_input or 'premium' in user_input:
            self.user_preferences['budget'] = 'luxury'
        
        if 'family' in user_input:
            self.user_preferences['group_type'] = 'family'
        elif 'solo' in user_input or 'alone' in user_input:
            self.user_preferences['group_type'] = 'solo'
        
        if 'temple' in user_input or 'religious' in user_input:
            self.user_preferences['interests'] = self.user_preferences.get('interests', []) + ['temples']
        if 'food' in user_input or 'cuisine' in user_input:
            self.user_preferences['interests'] = self.user_preferences.get('interests', []) + ['food']
    
    def get_personalized_greeting(self) -> str:
        base_greeting = "**Welcome to Saanchari!**\n\nYour comprehensive Andhra Pradesh tourism guide."
        
        if self.user_preferences.get('budget') == 'budget':
            base_greeting += " I'll focus on budget-friendly options for you."
        elif self.user_preferences.get('budget') == 'luxury':
            base_greeting += " I'll suggest premium experiences for you."
        
        if 'temples' in self.user_preferences.get('interests', []):
            base_greeting += " I see you're interested in temples - AP has amazing spiritual destinations!"
        
        base_greeting += "\n\nWhat would you like to explore in Andhra Pradesh?"
        return base_greeting
    
    def is_followup_question(self, text: str) -> bool:
        followup_indicators = ['more', 'tell me more', 'what about', 'also', 'and', 'other', 'else', 'similar']
        return any(indicator in text for indicator in followup_indicators) and len(self.conversation_history) > 1
    
    def handle_followup(self, user_input: str) -> str:
        """Handle follow-up questions based on conversation context"""
        if not self.conversation_history:
            return self.get_intelligent_fallback()
        
        last_query = self.conversation_history[-2] if len(self.conversation_history) > 1 else ""
        
        if 'temple' in last_query:
            return "Here are more temples you might be interested in:\n\n" + self.kb.get_temple_info("temples")
        elif 'food' in last_query:
            return "Here are more food recommendations:\n\n" + self.kb.get_food_info("cuisine")
        elif any(dest in last_query for dest in ['tirupati', 'visakhapatnam', 'araku']):
            return "Here are other popular destinations:\n\n" + self.get_top_destinations()
        
        return self.process_intelligent_query(user_input)
    
    def process_intelligent_query(self, query: str) -> str:
        """Enhanced query processing with advanced intelligence"""
        
        # Handle comparison queries specially
        if any(word in query.lower() for word in ['vs', 'versus', 'compare', 'difference']):
            comparison_response = self.handle_comparison_queries(query)
            if comparison_response:
                return comparison_response
        
        # Use the enhanced smart search
        search_results = self.kb.smart_search(query, max_results=2)
        
        if not search_results:
            return ""
        
        # Format the best result with intelligence
        best_result = search_results[0]
        response = self.kb.format_search_result(best_result)
        
        # Add contextual information based on user preferences
        if self.user_preferences.get('budget') == 'budget':
            response += "\n\n*💡 Budget Tip: Look for APSRTC buses and government accommodations for affordable travel.*"
        elif self.user_preferences.get('budget') == 'luxury':
            response += "\n\n*✨ Luxury Tip: Consider premium hotels and private transport for a comfortable experience.*"
        
        # Add contextual help
        response += self.provide_contextual_help(query)
        
        # Generate smart suggestions
        suggestions = self.generate_smart_suggestions(response, query)
        if suggestions:
            response += "\n\n**You might also ask:**\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
        
        return response
    
    def add_personalized_suggestions(self, response: str) -> str:
        """Add personalized suggestions based on user preferences"""
        suggestions = []
        
        if 'tirupati' in response.lower() and 'temples' in self.user_preferences.get('interests', []):
            suggestions.append("*You might also like Simhachalam Temple in Visakhapatnam*")
        
        if 'food' in response.lower() and self.user_preferences.get('group_type') == 'family':
            suggestions.append("*Family-friendly restaurants usually have mild spice options*")
        
        if suggestions:
            response += "\n\n**Personalized Suggestions:**\n" + "\n".join(suggestions)
        
        return response
    
    def get_intelligent_fallback(self) -> str:
        """Intelligent fallback based on conversation history"""
        if 'temple' in ' '.join(self.conversation_history[-3:]):
            return """**I can help you with temple information!**

Try asking:
• "Tell me about Tirupati temple timings"
• "Famous temples in Andhra Pradesh"
• "Temple etiquette and dress code"

*What specific temple information do you need?*"""
        
        elif 'food' in ' '.join(self.conversation_history[-3:]):
            return """**I can help you with Andhra Pradesh cuisine!**

Try asking:
• "Best biryani restaurants in Hyderabad"
• "Traditional Andhra dishes"
• "Spicy food recommendations"

*What food information interests you?*"""
        
        return """**I'm here to help with Andhra Pradesh tourism!**

Based on our conversation, I can provide information about:

**Destinations:** Tirupati, Visakhapatnam, Araku Valley, Vijayawada, and more
**Temples:** Tirumala, Simhachalam, Kanaka Durga, Srisailam
**Food:** Hyderabadi Biryani, Andhra meals, local specialties
**Travel:** Transportation, routes, accommodation, safety tips

*Please ask me about any specific place, food, temple, or travel information you need!*"""

    def load_question_patterns(self) -> dict:
        """Load common question patterns for better understanding"""
        return {
            'comparison': ['vs', 'versus', 'compare', 'difference between', 'better than', 'which is better'],
            'recommendation': ['best', 'top', 'recommend', 'suggest', 'should i', 'which one'],
            'planning': ['plan', 'itinerary', 'schedule', 'organize', 'arrange'],
            'cost': ['cost', 'price', 'budget', 'expensive', 'cheap', 'how much', 'money'],
            'time': ['when', 'time', 'duration', 'how long', 'best time', 'season'],
            'location': ['where', 'location', 'address', 'situated', 'located'],
            'how_to': ['how to', 'how can', 'way to', 'method', 'process']
        }
    
    def detect_question_type(self, query: str) -> str:
        """Detect the type of question being asked"""
        query_lower = query.lower()
        
        for question_type, patterns in self.question_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return question_type
        
        return 'general'
    
    def generate_smart_suggestions(self, current_response: str, query: str) -> list:
        """Generate intelligent follow-up suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        # If asking about a destination, suggest related queries
        if 'tirupati' in query_lower:
            suggestions.extend([
                "How to book darshan tickets online?",
                "Best time to visit Tirupati temple?",
                "Accommodation near Tirumala?"
            ])
        elif 'visakhapatnam' in query_lower or 'vizag' in query_lower:
            suggestions.extend([
                "Best beaches in Visakhapatnam?",
                "Araku Valley train journey details?",
                "Water sports in Vizag beaches?"
            ])
        elif 'food' in query_lower or 'biryani' in query_lower:
            suggestions.extend([
                "Vegetarian restaurants in Hyderabad?",
                "Traditional Andhra breakfast items?",
                "Food safety tips for tourists?"
            ])
        elif 'temple' in query_lower:
            suggestions.extend([
                "Temple dress code guidelines?",
                "Photography rules in temples?",
                "Temple festival calendar?"
            ])
        
        # Add contextual suggestions based on question type
        question_type = self.detect_question_type(query)
        if question_type == 'cost' and not any('budget' in s.lower() for s in suggestions):
            suggestions.append("Budget travel tips for AP?")
        elif question_type == 'time' and not any('season' in s.lower() for s in suggestions):
            suggestions.append("Weather patterns in different seasons?")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def handle_comparison_queries(self, query: str) -> str:
        """Handle comparison questions intelligently"""
        query_lower = query.lower()
        
        # Common comparisons
        if 'tirupati vs' in query_lower or 'tirupati and' in query_lower:
            return """**Tirupati vs Other Destinations:**

**Tirupati** - Best for spiritual/religious tourism
• World's richest temple
• Massive crowds, advance booking needed
• 2-3 days sufficient

**Visakhapatnam** - Best for beaches and relaxation  
• Coastal city with beaches and hills
• Less crowded, more leisure activities
• 3-4 days recommended

**Araku Valley** - Best for nature and coffee lovers
• Hill station with scenic beauty
• Tribal culture and coffee plantations
• 2-3 days ideal

*Which type of experience interests you more?*"""
        
        elif 'hyderabad vs' in query_lower:
            return """**Hyderabad Comparison:**

**For History & Culture:** Hyderabad wins
• Rich Nizami heritage, Charminar, Golconda Fort

**For Beaches:** Visakhapatnam is better
• Coastal location with multiple beaches

**For Spirituality:** Tirupati is unmatched
• Major pilgrimage destination

**For Food:** Hyderabad is famous
• Authentic Hyderabadi Biryani birthplace

*What aspect interests you most for comparison?*"""
        
        return ""
    
    def provide_contextual_help(self, query: str) -> str:
        """Provide contextual help based on query complexity"""
        if len(query.split()) < 3:
            return "\n\n*💡 Tip: Try asking more specific questions like 'Best time to visit Tirupati temple' for detailed information.*"
        
        question_type = self.detect_question_type(query)
        
        if question_type == 'cost':
            return "\n\n*💰 Cost Tip: Prices vary by season. October-March is peak season with higher rates.*"
        elif question_type == 'time':
            return "\n\n*🕐 Timing Tip: Check local festival calendars as they affect crowd levels and accommodation availability.*"
        elif question_type == 'planning':
            return "\n\n*📋 Planning Tip: Use our Itinerary Generator tab for detailed trip planning with cost breakdown.*"
        
        return ""
    
    def format_ai_response(self, ai_response: dict, original_query: str) -> str:
        """Format AI response with metadata and enhancements"""
        response = ai_response["answer"]
        
        # Add source indicator
        if ai_response["source"] == "AI + Knowledge Base":
            response = f"🤖 **AI Enhanced Response:**\n\n{response}"
        elif ai_response["source"] == "Enhanced Knowledge Base":
            response = f"📚 **Knowledge Base Response:**\n\n{response}"
        elif ai_response["source"] == "Offline AI":
            response = f"🔧 **Smart Response:**\n\n{response}"
        
        # Add confidence indicator for AI responses
        if ai_response["confidence"] > 0.7:
            confidence_emoji = "🎯"
        elif ai_response["confidence"] > 0.4:
            confidence_emoji = "📊"
        else:
            confidence_emoji = "💡"
        
        # Add sentiment-based personalization
        sentiment = ai_response.get("sentiment", "neutral")
        if sentiment == "negative":
            response = "I understand you might need help with this. Let me assist you.\n\n" + response
        elif sentiment == "positive":
            response = "Great to see your enthusiasm! " + response
        
        # Add smart suggestions if available
        suggestions = ai_response.get("suggestions", [])
        if suggestions:
            response += "\n\n**💡 You might also ask:**\n"
            for i, suggestion in enumerate(suggestions, 1):
                response += f"{i}. {suggestion}\n"
        
        # Add debug info in sidebar (for development)
        if st.sidebar.checkbox("Show AI Debug Info", key="ai_debug"):
            st.sidebar.write(f"**Source:** {ai_response['source']}")
            st.sidebar.write(f"**Confidence:** {ai_response['confidence']:.2f}")
            st.sidebar.write(f"**Sentiment:** {sentiment}")
        
        return response
    
    def get_fallback_response(self) -> str:
        return self.get_intelligent_fallback()

class ItineraryGenerator:
    def __init__(self):
        self.itinerary_templates = {
            "spiritual_tour": {
                "name": "Spiritual & Temple Tour",
                "duration_options": [3, 5, 7, 10],
                "main_destinations": ["Tirupati", "Srisailam", "Vijayawada", "Amaravati"],
                "theme": "Religious and spiritual experiences",
                "budget_range": "₹15,000 - ₹40,000 per person"
            },
            "cultural_heritage": {
                "name": "Cultural & Heritage Tour",
                "duration_options": [5, 7, 10, 14],
                "main_destinations": ["Hyderabad", "Warangal", "Gandikota", "Lepakshi"],
                "theme": "Historical monuments and cultural sites",
                "budget_range": "₹20,000 - ₹50,000 per person"
            },
            "coastal_adventure": {
                "name": "Coastal & Beach Tour",
                "duration_options": [4, 6, 8],
                "main_destinations": ["Visakhapatnam", "Araku Valley", "Machilipatnam"],
                "theme": "Beaches, water activities, and coastal culture",
                "budget_range": "₹18,000 - ₹45,000 per person"
            }
        }
    
    def create_custom_itinerary(self, preferences: Dict) -> Dict:
        duration = preferences.get('duration', 7)
        theme = preferences.get('theme', 'spiritual_tour')
        
        template = self.itinerary_templates.get(theme, self.itinerary_templates['spiritual_tour'])
        destinations = template['main_destinations'][:duration//2 + 1]
        
        itinerary = {}
        
        for day in range(1, duration + 1):
            day_key = f"Day {day}"
            
            if day == 1:
                itinerary[day_key] = {
                    "location": destinations[0],
                    "activities": [
                        "Arrival and check-in",
                        "Local orientation tour",
                        "Welcome dinner with traditional cuisine"
                    ],
                    "meals": "Breakfast, Lunch, Dinner"
                }
            elif day == duration:
                itinerary[day_key] = {
                    "location": destinations[-1],
                    "activities": [
                        "Final sightseeing",
                        "Shopping for souvenirs",
                        "Departure"
                    ],
                    "meals": "Breakfast, Lunch"
                }
            else:
                dest_index = min((day - 1) // 2, len(destinations) - 1)
                current_dest = destinations[dest_index]
                
                activities = self.get_destination_activities(current_dest, theme)
                
                itinerary[day_key] = {
                    "location": current_dest,
                    "activities": activities,
                    "meals": "Breakfast, Lunch, Dinner"
                }
        
        return itinerary
    
    def get_destination_activities(self, destination: str, theme: str) -> List[str]:
        base_activities = {
            "Tirupati": [
                "Tirumala Venkateswara Temple darshan",
                "Sri Padmavathi Ammavari Temple visit",
                "TTD Gardens exploration"
            ],
            "Visakhapatnam": [
                "RK Beach morning walk",
                "Submarine Museum visit",
                "Kailasagiri Hill Park"
            ],
            "Hyderabad": [
                "Charminar and Chowmahalla Palace",
                "Golconda Fort exploration",
                "Salar Jung Museum"
            ],
            "Araku Valley": [
                "Coffee plantation tour",
                "Tribal Museum visit",
                "Borra Caves exploration"
            ]
        }
        
        activities = base_activities.get(destination, ["Local sightseeing", "Cultural exploration", "Traditional dining"])
        return activities[:3]
    
    def generate_itinerary_flowchart(self, itinerary_data: Dict) -> go.Figure:
        fig = go.Figure()
        
        days = list(itinerary_data.keys())
        y_positions = list(range(len(days), 0, -1))
        
        # Create nodes for each day
        for i, (day, details) in enumerate(itinerary_data.items()):
            # Main day node
            fig.add_trace(go.Scatter(
                x=[1], y=[y_positions[i]],
                mode='markers+text',
                marker=dict(size=40, color='lightblue', line=dict(width=2, color='darkblue')),
                text=day,
                textposition="middle center",
                textfont=dict(size=12, color='darkblue'),
                name=day,
                hovertemplate=f"<b>{day}</b><br>{details.get('location', '')}<extra></extra>"
            ))
            
            # Activity nodes
            activities = details.get('activities', [])
            for j, activity in enumerate(activities[:3]):
                fig.add_trace(go.Scatter(
                    x=[2 + j * 0.8], y=[y_positions[i]],
                    mode='markers+text',
                    marker=dict(size=25, color='lightgreen', line=dict(width=1, color='darkgreen')),
                    text=activity[:15] + "..." if len(activity) > 15 else activity,
                    textposition="middle center",
                    textfont=dict(size=8),
                    showlegend=False,
                    hovertemplate=f"<b>Activity:</b> {activity}<extra></extra>"
                ))
                
                # Connect day to activities
                fig.add_trace(go.Scatter(
                    x=[1.2, 2 + j * 0.8], y=[y_positions[i], y_positions[i]],
                    mode='lines',
                    line=dict(color='gray', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Connect days with arrows
        for i in range(len(y_positions) - 1):
            fig.add_annotation(
                x=1, y=y_positions[i] - 0.3,
                ax=1, ay=y_positions[i + 1] + 0.3,
                arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor='blue'
            )
        
        fig.update_layout(
            title="Itinerary Flowchart",
            xaxis=dict(showgrid=False, showticklabels=False, range=[0.5, 5]),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False,
            height=100 * len(days) + 200,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def get_cost_breakdown(self, itinerary: Dict, preferences: Dict) -> Dict:
        duration = len(itinerary)
        budget_level = preferences.get('budget', 'medium')
        group_size = preferences.get('group_size', 2)
        
        cost_per_day = {
            "budget": {"accommodation": 1500, "food": 800, "transport": 600, "activities": 500},
            "medium": {"accommodation": 3000, "food": 1200, "transport": 1000, "activities": 800},
            "luxury": {"accommodation": 6000, "food": 2000, "transport": 1500, "activities": 1200}
        }
        
        daily_cost = cost_per_day.get(budget_level, cost_per_day["medium"])
        
        total_cost = {
            "accommodation": daily_cost["accommodation"] * duration * group_size,
            "food": daily_cost["food"] * duration * group_size,
            "transport": daily_cost["transport"] * duration * group_size,
            "activities": daily_cost["activities"] * duration * group_size
        }
        
        total_cost["total"] = sum(total_cost.values())
        return total_cost

# Initialize components with better caching
@st.cache_resource
def load_chat_engine():
    return ChatEngine()

@st.cache_resource  
def load_itinerary_generator():
    return ItineraryGenerator()

# Clear cache if needed
if st.sidebar.button("🔄 Clear Cache", help="Clear cached responses for fresh answers"):
    st.cache_resource.clear()
    st.rerun()

chat_engine = load_chat_engine()
itinerary_gen = load_itinerary_generator()

# Language mapping (display only - no translation for now)
LANGUAGES = {
    'English': 'en',
    'हिंदी': 'hi', 
    'తెలుగు': 'te'
}

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Language selector (currently English only - translation coming soon)
col1, col2, col3 = st.columns([6, 1, 1])
with col3:
    st.info("🌐 English")
    # Translation feature will be added in future updates

# Simple translation function (disabled for now - using English only)
def translate_text(text: str, target_lang: str) -> str:
    # For now, return text as-is (English only)
    # Translation can be added later with a more stable library
    return text

# Title and description
st.title("Saanchari")
st.subheader("Your Comprehensive Andhra Pradesh Tourism Assistant")

# Welcome message
welcome_message = "Welcome to Saanchari! I'm your dedicated Andhra Pradesh tourism assistant. Ask me anything about destinations, food, temples, travel routes, accommodations, and itineraries in AP!"

st.info(welcome_message)

# Add navigation tabs
tab1, tab2 = st.tabs(["💬 Chat Assistant", "🗺️ Itinerary Generator"])

with tab1:
    # Quick suggestion buttons
    st.subheader("🚀 Quick Questions:")
    quick_questions = [
        "Top 10 destinations in Andhra Pradesh",
        "Famous temples to visit",
        "Traditional Andhra cuisine",
        "How to reach Tirupati?",
        "Best beaches in AP",
        "Hyderabadi Biryani restaurants"
    ]

    cols = st.columns(3)
    for i, question in enumerate(quick_questions):
        col_idx = i % 3
        with cols[col_idx]:
            if st.button(question, key=f"quick_{i}"):
                st.session_state.messages.append({"role": "user", "content": question})
                response = chat_engine.get_response(question)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    # Chat interface
    st.subheader("💬 Chat with Saanchari:")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about Andhra Pradesh tourism..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.get_response(prompt)
            
            # Display the response
            st.markdown(response)
            
            # Add to session state
            st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    # Itinerary Generator Interface
    st.header("🗺️ Custom Itinerary Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("Trip Duration (days)", 3, 15, 7)
        theme = st.selectbox("Tour Theme", list(itinerary_gen.itinerary_templates.keys()))
        budget = st.selectbox("Budget Level", ["budget", "medium", "luxury"])
        
    with col2:
        group_size = st.number_input("Group Size", 1, 10, 2)
        group_type = st.selectbox("Group Type", ["family", "friends", "couple", "solo"])
        start_date = st.date_input("Start Date", datetime.now().date())
    
    if st.button("Generate Custom Itinerary", type="primary"):
        preferences = {
            'duration': duration,
            'theme': theme,
            'budget': budget,
            'group_size': group_size,
            'group_type': group_type,
            'start_date': start_date
        }
        
        # Generate itinerary
        custom_itinerary = itinerary_gen.create_custom_itinerary(preferences)
        
        # Display itinerary
        st.subheader("📋 Your Custom Itinerary")
        
        # Flowchart
        st.subheader("🔄 Itinerary Flowchart")
        flowchart = itinerary_gen.generate_itinerary_flowchart(custom_itinerary)
        st.plotly_chart(flowchart, use_container_width=True)
        
        # Detailed itinerary
        for day, details in custom_itinerary.items():
            with st.expander(f"📅 {day} - {details['location']}"):
                st.write(f"**Location:** {details['location']}")
                st.write("**Activities:**")
                for activity in details['activities']:
                    st.write(f"• {activity}")
                st.write(f"**Meals:** {details['meals']}")
        
        # Cost breakdown
        st.subheader("💰 Cost Breakdown")
        cost_breakdown = itinerary_gen.get_cost_breakdown(custom_itinerary, preferences)
        
        cost_col1, cost_col2 = st.columns(2)
        with cost_col1:
            st.metric("Accommodation", f"₹{cost_breakdown['accommodation']:,}")
            st.metric("Food & Dining", f"₹{cost_breakdown['food']:,}")
        with cost_col2:
            st.metric("Transportation", f"₹{cost_breakdown['transport']:,}")
            st.metric("Activities", f"₹{cost_breakdown['activities']:,}")
        
        st.metric("**Total Cost**", f"₹{cost_breakdown['total']:,}", help="Total cost for all travelers")

# Sidebar with additional information
with st.sidebar:
    st.header("🏛️ About Saanchari")
    st.markdown("""
    **Saanchari** is your comprehensive guide to Andhra Pradesh tourism. 
    
    **Features:**
    - 🗺️ Complete destination guide
    - 🍽️ Food and cuisine information
    - 🏛️ Temple and heritage sites
    - 🚗 Travel routes and transportation
    - 📅 Customized itineraries with flowcharts
   
   
    """)
    
    st.header("Contact:+919035235665")
    st.markdown("""
    
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
