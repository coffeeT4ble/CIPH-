package com.example.ciph

data class ParkingResult(
    val parkingId: String,
    val hour: Int,
    val day: String,
    val occupancyRate: Float,
    val occupied: Int,
    val free: Int,
    val total: Int,
    val status: String,
    val isObserved: Boolean
)