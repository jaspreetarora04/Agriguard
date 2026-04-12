#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

#include "camera_pins.h"   // ← humne manually banayi hai

/* ---------- WIFI DETAILS ---------- */
const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";

/* ---------- FLASK SERVER URL ---------- */
String serverName = "http://YOUR/predict";
#define PUMP_PIN 12

void setup() {
  Serial.begin(115200);
  Serial.println("Starting ESP32-CAM...");
  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, LOW);

  /* ===== WIFI CONNECT ===== */
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");
  Serial.print("ESP IP: ");
  Serial.println(WiFi.localIP());
  /* ===== CAMERA INIT ===== */
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;

  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size   = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count     = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }

  Serial.println("Camera initialized successfully");
  }
void loop() {

  if (WiFi.status() == WL_CONNECTED) {

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      delay(2000);
      return;
    }

    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "image/jpeg");

    int httpResponseCode = http.POST(fb->buf, fb->len);

    Serial.print("HTTP Response Code: ");
    Serial.println(httpResponseCode);
    if (httpResponseCode <= 0) {
  Serial.println("Error sending request");
}

    // ===== MOSFET / PUMP CONTROL =====
    if (httpResponseCode == 200) {

      String payload = http.getString();
      Serial.println(payload);

      if (payload.indexOf("Infected") != -1) {
        digitalWrite(PUMP_PIN, HIGH);   // Pump ON
        Serial.println(" Pump ON");
          delay(3000);   // spray 3 sec

  digitalWrite(PUMP_PIN, LOW);
      }
      else {
        digitalWrite(PUMP_PIN, LOW);    // Pump OFF
        Serial.println(" Pump OFF");
      }
    }

    http.end();
    esp_camera_fb_return(fb);
  }
  else {
    Serial.println("WiFi not connected");
  }

  delay(5000);   // 5 seconds gap between checks
}