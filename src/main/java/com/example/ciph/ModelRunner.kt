package com.example.ciph

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.roundToInt
import kotlin.math.roundToInt

class ModelRunner(context: Context) {

    // Mirrors DAY_MAP in model.py
    private val dayMap = mapOf(
        "monday" to 0, "tuesday" to 1, "wednesday" to 2, "thursday" to 3,
        "friday" to 4, "saturday" to 5, "sunday" to 6,
        "mon" to 0, "tue" to 1, "wed" to 2, "thu" to 3,
        "fri" to 4, "sat" to 5, "sun" to 6
    )
    private val dayNames = listOf(
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    )

    // Loaded from parking_id_info.json (replaces LabelEncoder + id_info in model.py)
    private val idInfo: Map<String, IdInfo>
    private val labelEncoding: Map<String, Int>   // parking_id → integer index (sorted order)
    private val ortEnv: OrtEnvironment
    private val session: OrtSession

    data class IdInfo(
        val totalSpots: Int,
        val source: String,
        val meanOccupancy: Float,
        val hourlyActuals: Map<Int, Float>   // hour → observed rate
    )

    init {
        // ── Load ONNX model from assets ──────────────────────────────────────
        ortEnv = OrtEnvironment.getEnvironment()
        val modelBytes = context.assets.open("parking_model.onnx").readBytes()
        session = ortEnv.createSession(modelBytes, OrtSession.SessionOptions())

        // ── Load id info JSON from assets ────────────────────────────────────
        val jsonStr = context.assets.open("parking_id_info.json")
            .bufferedReader().readText()
        val root = JSONObject(jsonStr)

        // Build idInfo map
        val rawMap = mutableMapOf<String, IdInfo>()
        root.keys().forEach { pid ->
            val obj = root.getJSONObject(pid)
            val actualsObj = obj.optJSONObject("hourly_actuals") ?: JSONObject()
            val actuals = mutableMapOf<Int, Float>()
            actualsObj.keys().forEach { h ->
                actuals[h.toInt()] = actualsObj.getDouble(h).toFloat()
            }
            rawMap[pid] = IdInfo(
                totalSpots    = obj.getInt("total_spots"),
                source        = obj.getString("source"),
                meanOccupancy = obj.getDouble("mean_occupancy").toFloat(),
                hourlyActuals = actuals
            )
        }
        idInfo = rawMap

        // Reproduce LabelEncoder: sorted alphabetical → index
        // This MUST match how sklearn's LabelEncoder assigned integers during training
        labelEncoding = idInfo.keys.sorted()
            .mapIndexed { idx, pid -> pid to idx }
            .toMap()
    }

    // ── Public predict(), mirrors model.py predict() exactly ─────────────────
    fun predict(parkingId: String, hour: Int, day: String): ParkingResult {

        // Resolve parking ID (exact or fuzzy match)
        val resolvedId = resolveId(parkingId)
            ?: throw IllegalArgumentException(
                "ID '$parkingId' not found. Available: ${idInfo.keys.take(5)}"
            )

        require(hour in 0..23) { "hour must be 0–23" }

        val dayKey = day.lowercase()
        val dow = dayMap[dayKey]
            ?: throw IllegalArgumentException("Unrecognised day '$day'")

        val info    = idInfo[resolvedId]!!
        val total   = info.totalSpots
        val meanOcc = info.meanOccupancy
        val encId   = labelEncoding[resolvedId]!!.toFloat()

        // ── Build feature vector: [parking_id_enc, hour, id_mean_occupancy, total_spot_number]
        // Order MUST match FEATURES in model.py
        val featureArray = floatArrayOf(
            encId,
            hour.toFloat(),
            meanOcc,
            total.toFloat()
        )

        // ── Run ONNX inference ────────────────────────────────────────────────
        val inputTensor = OnnxTensor.createTensor(
            ortEnv,
            FloatBuffer.wrap(featureArray),
            longArrayOf(1, 4)   // shape: [1 sample, 4 features]
        )

        val outputName = session.outputNames.first()
        val results = session.run(mapOf(session.inputNames.first() to inputTensor))
        val rawRate = (results[outputName].get().value as Array<FloatArray>)[0][0]
        val rate = rawRate.coerceIn(0f, 1f)

        inputTensor.close()
        results.close()

        // ── Derive outputs, same logic as model.py ───────────────────────────
        val occupied = (rate * total).roundToInt()
        val free     = total - occupied

        val isObserved  = info.hourlyActuals.containsKey(hour)

        val status = when {
            free == 0    -> "FULL"
            rate >= 0.85f -> "Nearly full"
            rate >= 0.5f  -> "Moderately busy"
            else          -> "Plenty of space"
        }

        return ParkingResult(
            parkingId    = resolvedId,
            hour         = hour,
            day          = dayNames[dow],
            occupancyRate = rate,
            occupied     = occupied,
            free         = free,
            total        = total,
            status       = status,
            isObserved   = isObserved
        )
    }

    fun listIds(): List<String> = idInfo.keys.sorted()

    // Fuzzy match: exact first, then substring (mirrors model.py)
    private fun resolveId(input: String): String? {
        if (idInfo.containsKey(input)) return input
        val matches = idInfo.keys.filter { input.lowercase() in it.lowercase() }
        return if (matches.size == 1) matches[0] else null
    }

    fun close() {
        session.close()
        ortEnv.close()
    }
}