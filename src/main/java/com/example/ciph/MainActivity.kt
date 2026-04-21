package com.example.ciph

import android.os.Bundle
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.example.ciph.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var runner: ModelRunner

    private val days = listOf(
        "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Load model on background thread (ONNX init reads from assets)
        lifecycleScope.launch(Dispatchers.IO) {
            runner = ModelRunner(this@MainActivity)
            val ids = runner.listIds()

            withContext(Dispatchers.Main) {
                setupSpinners(ids)
            }
        }

        // Hour picker: 0–23
        binding.pickerHour.minValue = 0
        binding.pickerHour.maxValue = 23
        binding.pickerHour.value    = 9

        binding.btnPredict.setOnClickListener { runPrediction() }
    }

    private fun setupSpinners(ids: List<String>) {
        binding.spinnerParkingId.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, ids)

        binding.spinnerDay.adapter =
            ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, days)
        binding.spinnerDay.setSelection(0) // Monday default
    }

    private fun runPrediction() {
        val parkingId = binding.spinnerParkingId.selectedItem?.toString() ?: return
        val hour      = binding.pickerHour.value
        val day       = binding.spinnerDay.selectedItem?.toString() ?: return

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val result = runner.predict(parkingId, hour, day)
                withContext(Dispatchers.Main) { displayResult(result) }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, e.message, Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun displayResult(r: ParkingResult) {
        val pct = (r.occupancyRate * 100).toInt()

        binding.tvStatus.text    = "${statusIcon(r.status)}  ${r.status}"
        binding.tvOccupancy.text = "Occupied: ${r.occupied} / ${r.total} spots  ($pct%)"
        binding.tvTotal.text     = "Free: ${r.free} spots"
        binding.tvObserved.text  = if (r.isObserved)
            "✓ Based on observed data at ${r.hour}:00"
        else
            "⚠ Extrapolated — no data yet for this hour"

        binding.progressOccupancy.progress = pct
    }

    private fun statusIcon(status: String) = when (status) {
        "FULL"             -> "🔴"
        "Nearly full"      -> "🟠"
        "Moderately busy"  -> "🟡"
        else               -> "🟢"
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::runner.isInitialized) runner.close()
    }
}