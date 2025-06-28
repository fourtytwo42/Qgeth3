package main

import (
	"context"
	"embed"
	"fmt"
	"log"
	"os"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

//go:embed all:frontend/dist
var assets embed.FS

// App struct
type App struct {
	ctx           context.Context
	walletService *WalletService
}

// NewApp creates a new App application struct
func NewApp() *App {
	return &App{}
}

// OnStartup is called when the app starts up
func (a *App) OnStartup(ctx context.Context) {
	a.ctx = ctx
	a.walletService = NewWalletService()
	
	// Initialize wallet service
	if err := a.walletService.Initialize(); err != nil {
		log.Printf("Failed to initialize wallet service: %v", err)
	}
}

// OnShutdown is called when the app is shutting down
func (a *App) OnShutdown(ctx context.Context) {
	if a.walletService != nil {
		a.walletService.Shutdown()
	}
}

// Greet returns a greeting for the given name
func (a *App) Greet(name string) string {
	return fmt.Sprintf("Hello %s, It's show time!", name)
}

func main() {
	// Create an instance of the app structure
	app := NewApp()

	// Create application with options
	err := wails.Run(&options.App{
		Title:         "Quantum Wallet",
		Width:         1200,
		Height:        800,
		MinWidth:      800,
		MinHeight:     600,
		Fullscreen:    false,
		StartHidden:   false,
		BackgroundColour: &options.RGBA{R: 27, G: 38, B: 54, A: 1},
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		OnStartup:  app.OnStartup,
		OnShutdown: app.OnShutdown,
		Bind: []interface{}{
			app,
		},
	})

	if err != nil {
		fmt.Printf("Error starting Quantum Wallet: %v\n", err)
		os.Exit(1)
	}
} 