from fastapi import APIRouter
from flightBookingOps import check_if_pnr_exists, get_pnr_details, cancel_ticket

router = APIRouter()

@router.get("/check_pnr/{pnr_number}")
def check_pnr(pnr_number: str):
    return {"exists": check_if_pnr_exists(pnr_number)}

@router.get("/pnr_details/{pnr_number}")
def fetch_pnr_details(pnr_number: str):
    return {"details": get_pnr_details(pnr_number)}

@router.delete("/cancel_ticket/{pnr_number}")
def cancel_flight_ticket(pnr_number: str):
    return {"message": cancel_ticket(pnr_number)}