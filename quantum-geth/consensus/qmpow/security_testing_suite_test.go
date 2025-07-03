package qmpow

import "testing"

func TestSecurityTestingSuite_Creation(t *testing.T) {
config := DefaultSecurityTestConfig()
suite := NewSecurityTestingSuite(config)

if suite == nil {
t.Fatal("Failed to create security testing suite")
}
}
