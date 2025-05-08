from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, EventType
from rasa_sdk.events import UserUtteranceReverted
from rasa_sdk.events import ConversationPaused

import psycopg2
import requests
import pymysql


# # ✅ PostgreSQL connection config
# DB_CONFIG_PostGres = {
#     "host": "localhost",
#     "database": "rasa_chatbot",  # 🔁 Replace with your DB name
#     "user": "postgres",        # 🔁 Your DB user
#     "password": "satgur123",  # 🔁 Set real password
#     "port": "5432"
# }

# ✅ MySQL connection config
DB_CONFIG = {
    "host": "localhost",
    "database": "iisc_capstone",  # 🔁 Replace with your DB name
    "user": "root",        # 🔁 Your DB user
    "password": "root",  # 🔁 Set real password
    "port": 3306
}


class ActionDefaultFallback(Action):

    def name(self) -> Text:
        return "action_default_fallback"
    
    async def run(self,dispatcher: CollectingDispatcher,tracker: Tracker,domain: Dict[Text, Any],) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Sorry, I didn’t quite understand that. Can you rephrase?")
        return [UserUtteranceReverted()]


class ActionCheckPNRStatus(Action):
    def name(self) -> Text:
        return "action_check_pnr_status"
    
    def run(self, dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]) -> List[EventType]:

        pnr = tracker.get_slot("pnr")
        api_url = f"http://127.0.0.1:8000/action/check_pnr/{pnr}"

        try:
            response = requests.get(api_url)
            data = response.json()

            if data.get("exists") == "true":
                dispatcher.utter_message(text=f"Here are your PNR details:\n{data['details']}\nDo you want to cancel the ticket?")
                return [SlotSet("pnr_valid", True)]
            else:
                dispatcher.utter_message(text="The PNR you provided is not active or has expired.")
                return [SlotSet("pnr_valid", False)]
        except Exception as e:
            dispatcher.utter_message(text="There was an error checking your PNR. Please try again later.")
            return []

class ActionCancelBooking(Action):
    def name(self) -> Text:
        return "action_cancel_booking"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        dispatcher.utter_message(response="utter_cancellation_confirmed")
        return []



# ✅ Action 1: Fetch Booking Details
class ActionFetchBookingDetails(Action):
    def name(self) -> str:
        return "action_fetch_booking_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        pnr = tracker.get_slot("pnr")
        customer_id = tracker.get_slot("customer_id")

        try:
            connection = pymysql.connect(**DB_CONFIG)
            #connection = psycopg2.connect(**DB_CONFIG_PostGres) 
            cursor = connection.cursor()
            cursor.execute(
                "SELECT miles_balance,upgrade_eligible FROM bookings WHERE pnr = %s AND customer_id = %s",
                (pnr, customer_id)
            )

            row = cursor.fetchone()
            if row:
                miles, available = row
                dispatcher.utter_message(
                    text=f"You have {miles} miles. Business class upgrade requires 30,000 miles."
                )
                return [
                    SlotSet("miles_balance", miles),
                    SlotSet("upgrade_available", available)
                ]
            else:
                dispatcher.utter_message(text="Booking not found. Please check your PNR and customer ID.")

        except Exception as e:
            dispatcher.utter_message(text=f"Database error: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()

        return []

# ✅ Action 2: Offer Cash + Miles
class ActionOfferCashMiles(Action):
    def name(self) -> str:
        return "action_offer_cash_miles"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        dispatcher.utter_message(
            text="You can upgrade using ₹8000 + 20,000 miles. Would you like to proceed?"
        )
        return []

# ✅ Action: consult_airline_travel_policy
class ActionConsultAirlineTravelPolicy(Action):
    def name(self) -> str:
        return "consult_airline_travel_policy"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[EventType]:
            user_message = tracker.latest_message.get("text")
            print(f"Calling RAG API with question: {user_message}")
            api_url = "http://127.0.0.1:8000/rag/rag_query"
            payload = {"question": user_message}
            try:
                response = requests.post(api_url, json=payload, headers={"Content-Type": "application/json"})
                response.raise_for_status()
                data = response.json()

                # Example: RAG API returns 'answer'
                if "response" in data:
                    dispatcher.utter_message(text=data["response"])
                else:
                    dispatcher.utter_message(text="I couldn't find a relevant answer. Can you rephrase your question?")                    
            except Exception as e:
                print(f"Error calling RAG API: {e}")
                dispatcher.utter_message(text="There was an error fetching the answer. Please try again later.")
            return []

# ✅ Action 3: Confirm and Store Upgrade
class ActionConfirmUpgrade(Action):
    def name(self) -> str:
        return "action_confirm_upgrade"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        pnr = tracker.get_slot("pnr")
        customer_id = tracker.get_slot("customer_id")
        upgrade_class = tracker.get_slot("upgrade_class") or "Business"
        miles_balance = tracker.get_slot("miles_balance") or 0

        try:
            connection = pymysql.connect(**DB_CONFIG)
            #connection = psycopg2.connect(**DB_CONFIG_PostGres)
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO upgrade_requests (pnr, customer_id, upgrade_class, miles_used)
                VALUES (%s, %s, %s, %s)
            """, (pnr, customer_id, upgrade_class, miles_balance))

            connection.commit()
            dispatcher.utter_message(text="✅ Your upgrade has been confirmed and logged.")
        except Exception as e:
            dispatcher.utter_message(text=f"❌ Failed to confirm upgrade: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()

        return []
    
# ✅ Action : Book Flight
class BookFlight(Action):
    def name(self) -> str:
        return "action_book_flight"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:
        print("\n--- Flight Booking Slots ---")
        departure_city = tracker.get_slot("departure_city")
        arrival_city = tracker.get_slot("arrival_city")
        travel_date = tracker.get_slot("travel_date")
        print(f"Departure City: {departure_city}")
        print(f"Arrival City: {arrival_city}")
        print(f"Travel Date: {travel_date}")

        dispatcher.utter_message(
            text=f"Great! Booking a flight from {departure_city} to {arrival_city} on {travel_date} ✈️. Please wait a moment while I confirm the details !!\n"
        )
        return []