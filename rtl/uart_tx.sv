`timescale 1ns / 1ps

module uart_tx #(
    parameter CLK_FREQ = 125_000_000,
    parameter BAUD_RATE = 115200
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [7:0] tx_data,
    input  logic tx_valid,
    output logic tx_busy,
    output logic tx
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    
    typedef enum logic [3:0] {
        IDLE, START, DATA0, DATA1, DATA2, DATA3, 
        DATA4, DATA5, DATA6, DATA7, STOP
    } state_t;
    
    state_t state;
    logic [15:0] clk_count;
    logic [7:0] data_reg;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            tx <= 1'b1;
            clk_count <= 0;
            tx_busy <= 0;
        end else begin
            case (state)
                IDLE: begin
                    tx <= 1'b1;
                    clk_count <= 0;
                    tx_busy <= 0;
                    
                    if (tx_valid) begin
                        data_reg <= tx_data;
                        state <= START;
                        tx_busy <= 1;
                    end
                end
                
                START: begin
                    tx <= 1'b0;  // Start bit
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        state <= DATA0;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                DATA0, DATA1, DATA2, DATA3, 
                DATA4, DATA5, DATA6, DATA7: begin
                    tx <= data_reg[state - DATA0];
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        clk_count <= 0;
                        state <= (state == DATA7) ? STOP : state_t'(state + 1);
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
                
                STOP: begin
                    tx <= 1'b1;  // Stop bit
                    if (clk_count == CLKS_PER_BIT - 1) begin
                        state <= IDLE;
                    end else begin
                        clk_count <= clk_count + 1;
                    end
                end
            endcase
        end
    end
    
endmodule
