package utils

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// FormatHashrate formats a hashrate value into a human-readable string
func FormatHashrate(hashrate float64) string {
	if hashrate < 1000 {
		return fmt.Sprintf("%.2f H/s", hashrate)
	} else if hashrate < 1000000 {
		return fmt.Sprintf("%.2f KH/s", hashrate/1000)
	} else if hashrate < 1000000000 {
		return fmt.Sprintf("%.2f MH/s", hashrate/1000000)
	} else {
		return fmt.Sprintf("%.2f GH/s", hashrate/1000000000)
	}
}

// FormatDuration formats a duration into a human-readable string
func FormatDuration(d time.Duration) string {
	if d.Hours() >= 24 {
		days := int(d.Hours() / 24)
		hours := int(d.Hours()) % 24
		return fmt.Sprintf("%dd %dh", days, hours)
	} else if d.Hours() >= 1 {
		hours := int(d.Hours())
		minutes := int(d.Minutes()) % 60
		return fmt.Sprintf("%dh %dm", hours, minutes)
	} else if d.Minutes() >= 1 {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return fmt.Sprintf("%dm %ds", minutes, seconds)
	} else {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
}

// FormatBytes formats byte count into human-readable string
func FormatBytes(bytes uint64) string {
	if bytes < 1024 {
		return fmt.Sprintf("%d B", bytes)
	} else if bytes < 1024*1024 {
		return fmt.Sprintf("%.2f KB", float64(bytes)/1024)
	} else if bytes < 1024*1024*1024 {
		return fmt.Sprintf("%.2f MB", float64(bytes)/(1024*1024))
	} else {
		return fmt.Sprintf("%.2f GB", float64(bytes)/(1024*1024*1024))
	}
}

// ParseHexUint64 parses a hex string to uint64
func ParseHexUint64(hex string) (uint64, error) {
	if len(hex) < 2 || hex[:2] != "0x" {
		return 0, fmt.Errorf("invalid hex format")
	}
	return strconv.ParseUint(hex[2:], 16, 64)
}

// FormatHexUint64 formats uint64 to hex string
func FormatHexUint64(val uint64) string {
	return "0x" + strconv.FormatUint(val, 16)
}

// HashData creates a SHA256 hash of the given data
func HashData(data []byte) []byte {
	hash := sha256.Sum256(data)
	return hash[:]
}

// HashString creates a SHA256 hash of the given string
func HashString(data string) string {
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// GetSystemInfo returns basic system information
func GetSystemInfo() map[string]interface{} {
	return map[string]interface{}{
		"os":          runtime.GOOS,
		"arch":        runtime.GOARCH,
		"cpu_count":   runtime.NumCPU(),
		"go_version":  runtime.Version(),
		"max_threads": runtime.GOMAXPROCS(0),
	}
}

// ValidateAddress validates an Ethereum address format
func ValidateAddress(address string) bool {
	if len(address) != 42 {
		return false
	}
	if !strings.HasPrefix(address, "0x") {
		return false
	}

	// Check if all characters after 0x are valid hex
	for _, char := range address[2:] {
		if !isHexChar(char) {
			return false
		}
	}

	return true
}

// isHexChar checks if a character is a valid hex character
func isHexChar(char rune) bool {
	return (char >= '0' && char <= '9') ||
		(char >= 'a' && char <= 'f') ||
		(char >= 'A' && char <= 'F')
}

// TruncateString truncates a string to the specified length with ellipsis
func TruncateString(str string, length int) string {
	if len(str) <= length {
		return str
	}
	if length <= 3 {
		return str[:length]
	}
	return str[:length-3] + "..."
}

// EnsureDirectoryExists ensures that a directory exists, creating it if necessary
func EnsureDirectoryExists(path string) error {
	// TODO: Implement directory creation
	return nil
}
